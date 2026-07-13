"""
ProposalGenerator: Generate ROI mask proposals for CAST.

This module provides different methods for generating mask proposals:
- MedSAM3ProposalGenerator: Uses MedSAM3 for medical image segmentation
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import cv2


class ProposalGenerator(ABC):
    """
    Abstract base class for generating mask proposals.
    """

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Generate mask proposals for the given image.

        Args:
            image: PIL Image (RGB)
            prompts: Optional list of text prompts to guide segmentation

        Returns:
            List of binary masks, each of shape (H, W)
        """
        pass

    def filter_proposals(
        self,
        masks: List[np.ndarray],
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.6,
        nms_threshold: float = 0.9,
        max_proposals: int = 50
    ) -> List[np.ndarray]:
        """
        Filter proposals based on area and apply NMS.

        Args:
            masks: List of binary masks
            min_area_ratio: Minimum area as ratio of image size
            max_area_ratio: Maximum area as ratio of image size
            nms_threshold: IoU threshold for NMS
            max_proposals: Maximum number of proposals to keep

        Returns:
            Filtered list of masks
        """
        if len(masks) == 0:
            return []

        H, W = masks[0].shape
        total_area = H * W
        min_area = min_area_ratio * total_area
        max_area = max_area_ratio * total_area

        # Filter by area
        filtered = []
        areas = []
        for mask in masks:
            area = mask.sum()
            if min_area <= area <= max_area:
                filtered.append(mask)
                areas.append(area)

        if len(filtered) == 0:
            return []

        # Apply NMS based on mask IoU
        keep_indices = self._mask_nms(filtered, areas, nms_threshold)
        filtered = [filtered[i] for i in keep_indices]

        # Limit number of proposals
        return filtered[:max_proposals]

    def _mask_nms(
        self,
        masks: List[np.ndarray],
        areas: List[float],
        threshold: float
    ) -> List[int]:
        """
        Apply NMS to masks based on IoU.

        Args:
            masks: List of binary masks
            areas: List of mask areas
            threshold: IoU threshold

        Returns:
            List of indices to keep
        """
        if len(masks) == 0:
            return []

        # Sort by area (larger first)
        order = np.argsort(areas)[::-1]

        keep = []
        suppressed = set()

        for i in order:
            if i in suppressed:
                continue

            keep.append(i)
            mask_i = masks[i].astype(bool)

            for j in order:
                if j in suppressed or j == i:
                    continue

                mask_j = masks[j].astype(bool)

                # Compute IoU
                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()
                iou = intersection / (union + 1e-8)

                if iou > threshold:
                    suppressed.add(j)

        return keep

    def filter_by_morphology(
        self,
        masks: List[np.ndarray],
        max_aspect_ratio: float = 10.0,
        max_connected_components: int = 5
    ) -> List[np.ndarray]:
        """
        Filter masks by morphological properties.

        Args:
            masks: List of binary masks
            max_aspect_ratio: Maximum aspect ratio of bounding box
            max_connected_components: Maximum number of connected components

        Returns:
            Filtered list of masks
        """
        filtered = []

        for mask in masks:
            # Check connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            if num_labels - 1 > max_connected_components:
                continue

            # Check aspect ratio
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
                if aspect_ratio > max_aspect_ratio:
                    continue

            filtered.append(mask)

        return filtered


class MedSAM3ProposalGenerator(ProposalGenerator):
    """
    Generate proposals using MedSAM3 (Medical SAM3 with LoRA).

    MedSAM3 is a text-guided medical image segmentation model that can
    segment medical images using medical concept text prompts.
    """

    def __init__(
        self,
        config_path: str,
        weights_path: Optional[str] = None,
        medsam3_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
        medical_concepts: Optional[List[str]] = None
    ):
        """
        Initialize MedSAM3 proposal generator.

        Args:
            config_path: Path to MedSAM3 config YAML
            weights_path: Path to LoRA weights (auto-detected if None)
            medsam3_path: Path to MedSAM3 repo directory (auto-detected if None)
            resolution: Input resolution for MedSAM3
            detection_threshold: Confidence threshold for detections
            nms_iou_threshold: IoU threshold for NMS
            device: Device to run on
            medical_concepts: List of medical concepts to use as prompts
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.medsam3_path = medsam3_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Default medical concepts — expanded to cover common organs and structures
        self.medical_concepts = medical_concepts or [
            # General
            "organ", "lesion", "tumor", "abnormality", "tissue", "nodule",
            # Thorax
            "lung", "heart", "trachea", "bronchi", "esophagus", "aorta",
            "rib", "chest wall", "mediastinum", "pleura", "diaphragm",
            # Abdomen
            "liver", "spleen", "kidney", "stomach", "pancreas", "gallbladder",
            "colon", "intestine", "bladder", "adrenal gland",
            # Head & Neck
            "brain", "skull", "eye", "sinus", "thyroid", "neck",
            # Musculoskeletal
            "bone", "spine", "spinal cord", "vertebra", "disc",
            "femur", "pelvis", "hip", "joint", "muscle",
            # Vascular
            "vessel", "artery", "vein",
        ]

        # Medical keyword vocabulary for question-aware prompt extraction
        self._medical_keywords = set(w.lower() for w in self.medical_concepts) | {
            "lung", "liver", "kidney", "heart", "brain", "bone", "spine",
            "spleen", "stomach", "bladder", "colon", "esophagus", "trachea",
            "pancreas", "gallbladder", "thyroid", "breast", "prostate",
            "uterus", "ovary", "aorta", "rib", "pelvis", "femur", "tibia",
            "skull", "hip", "shoulder", "knee", "ankle", "wrist", "neck",
            "chest", "abdomen", "head", "thorax", "vertebra", "disc",
            "bronchi", "intestine", "rectum", "appendix", "adrenal",
            "cornea", "retina", "optic nerve", "ventricle", "atrium",
            "tumor", "lesion", "nodule", "cyst", "fracture", "hemorrhage",
            "effusion", "edema", "inflammation", "calcification", "mass",
        }

        self.model = None
        self.transform = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of MedSAM3 model."""
        if self._initialized:
            return

        import sys
        import os
        import yaml

        # Add MedSAM3 to path
        medsam3_path = self.medsam3_path or os.path.join(os.path.dirname(__file__), '..', 'MedSAM3')
        if medsam3_path not in sys.path:
            sys.path.insert(0, medsam3_path)

        from sam3.model_builder import build_sam3_image_model
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        )
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Auto-detect weights if not provided
        if self.weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            self.weights_path = os.path.join(output_dir, 'best_lora_weights.pt')

        # Resolve SAM3 checkpoint path (prefer local to avoid gated HF download)
        ckpt_path = None
        env_ckpt = os.environ.get("SAM3_CKPT_PATH")
        if env_ckpt and os.path.exists(env_ckpt):
            ckpt_path = env_ckpt
        else:
            # Check common locations for SAM3 checkpoint
            for candidate in [
                os.path.join(os.path.dirname(self.config_path), '..', 'sam3.pt'),
                os.path.expanduser('~/.cache/sam3/sam3.pt'),
            ]:
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break

        # Build base model
        # Set default CUDA device so any internal .cuda() calls use the right GPU
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        print("Loading MedSAM3 model...")
        self.model = build_sam3_image_model(
            device=str(self.device),
            compile=False,
            load_from_HF=(ckpt_path is None),
            checkpoint_path=ckpt_path,
            bpe_path=os.path.join(medsam3_path, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            eval_mode=True
        )

        # Apply LoRA
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load weights
        if os.path.exists(self.weights_path):
            print(f"Loading LoRA weights from {self.weights_path}")
            load_lora_weights(self.model, self.weights_path)
        else:
            print(f"Warning: LoRA weights not found at {self.weights_path}")

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=self.resolution,
                    max_size=self.resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._initialized = True
        print("MedSAM3 model loaded successfully!")

    def extract_prompts_from_questions(self, questions: List[str]) -> List[str]:
        """
        Extract medical keywords from questions to use as targeted MedSAM3 prompts.

        Args:
            questions: List of question strings for this image

        Returns:
            Deduplicated list of medical keywords found in the questions
        """
        extracted = set()
        for q in questions:
            q_lower = q.lower()
            for kw in self._medical_keywords:
                if kw in q_lower:
                    extracted.add(kw)
        return list(extracted)

    def generate(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Generate mask proposals using MedSAM3.

        Combines all concepts into a single prompt for ONE forward pass.

        Args:
            image: PIL Image (RGB)
            prompts: Optional list of text prompts (uses medical_concepts if None)

        Returns:
            List of binary masks, each of shape (H, W)
        """
        self._lazy_init()

        prompts = prompts or self.medical_concepts
        # Combine all concepts into a single comma-separated prompt
        combined_prompt = ", ".join(prompts)

        with torch.no_grad():
            all_masks = self._predict_single_prompt(image, combined_prompt)

        # Filter proposals
        if len(all_masks) > 0:
            all_masks = self.filter_proposals(
                all_masks,
                min_area_ratio=0.001,
                max_area_ratio=0.6,
                nms_threshold=0.9
            )
            all_masks = self.filter_by_morphology(all_masks)

        return all_masks

    def _predict_single_prompt(
        self,
        image: Image.Image,
        prompt: str
    ) -> List[np.ndarray]:
        """
        Run MedSAM3 prediction for a single prompt.

        Returns list of masks for this prompt.
        """
        import sys
        import os
        medsam3_path = self.medsam3_path or os.path.join(os.path.dirname(__file__), '..', 'MedSAM3')
        if medsam3_path not in sys.path:
            sys.path.insert(0, medsam3_path)

        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
        )
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device

        w, h = image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=image,
            objects=[],
            size=[h, w]
        )

        # Create query
        query = FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )

        datapoint = Datapoint(find_queries=[query], images=[sam_image])
        datapoint = self.transform(datapoint)
        batch = collate_fn_api([datapoint], dict_key="input")["input"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(batch)

        # Extract masks
        masks = []
        last_output = outputs[-1]
        pred_logits = last_output['pred_logits']
        pred_masks = last_output.get('pred_masks', None)

        if pred_masks is not None:
            out_probs = pred_logits.sigmoid()
            scores = out_probs[0, :, :].max(dim=-1)[0]

            keep = scores > self.detection_threshold

            if keep.sum() > 0:
                masks_small = pred_masks[0, keep].sigmoid() > 0.5

                # Resize to original size
                masks_resized = F.interpolate(
                    masks_small.unsqueeze(0).float(),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) > 0.5

                masks = [m.cpu().numpy().astype(np.uint8) for m in masks_resized]

        return masks


class SimpleProposalGenerator(ProposalGenerator):
    """
    Simple proposal generator using basic image processing.

    Useful as a fallback when MedSAM3 is not available.
    Uses superpixel segmentation + thresholding.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 10.0
    ):
        """
        Args:
            n_segments: Number of superpixels
            compactness: SLIC compactness parameter
        """
        self.n_segments = n_segments
        self.compactness = compactness

    def generate(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Generate proposals using superpixel segmentation.
        """
        from skimage.segmentation import slic
        from skimage.measure import regionprops

        img_np = np.array(image)

        # Generate superpixels
        segments = slic(
            img_np,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0
        )

        # Convert each segment to a mask
        masks = []
        unique_segments = np.unique(segments)

        for seg_id in unique_segments:
            mask = (segments == seg_id).astype(np.uint8)
            masks.append(mask)

        # Also add merged adjacent segments
        # This helps create larger meaningful regions
        for seg_id in unique_segments:
            mask = (segments == seg_id).astype(np.uint8)
            # Dilate and find overlapping segments
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=2)

            overlapping = np.unique(segments[dilated > 0])
            if len(overlapping) > 1 and len(overlapping) <= 5:
                merged = np.isin(segments, overlapping).astype(np.uint8)
                masks.append(merged)

        # Filter proposals
        masks = self.filter_proposals(masks)

        return masks
