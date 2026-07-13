"""
MaskProvider: Abstract interface for providing ROI masks for CAST decoding.

This module defines the MaskProvider interface and several implementations:
- GTMaskProvider: Uses ground-truth masks (for upper bound experiments)
- NullMaskProvider: Returns empty masks (for baseline experiments)
- AutoMaskProvider: Automatically discovers ROI masks (mask-free setting)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import torch


class MaskProvider(ABC):
    """
    Abstract base class for providing ROI masks.

    All mask providers must implement get_mask() which returns
    a binary mask of shape (H, W) where ROI=1 and background=0.
    """

    @abstractmethod
    def get_mask(
        self,
        image: Image.Image,
        question: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Get ROI mask for the given image and question.

        Args:
            image: PIL Image (RGB)
            question: Question string
            meta: Optional metadata dict containing:
                - image_id: Unique identifier for the image
                - bbox: Bounding box (x1, y1, x2, y2) if available
                - answer: Ground truth answer (for closed-set scoring)
                - mask_path: Path to GT mask if available

        Returns:
            mask: Binary numpy array of shape (H, W), values in {0, 1}
                  ROI=1, background=0
        """
        pass

    def get_mask_value(self, mask: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> int:
        """
        Get the mask value for the ROI region.

        Args:
            mask: Binary mask array
            bbox: Optional bounding box

        Returns:
            Mask value (typically 1 for ROI, 0 for background)
        """
        if bbox is not None and bbox != (0.0, 0.0, 0.0, 0.0):
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                return int(mask[cy, cx])
        return 1 if mask.max() > 0 else 0


class GTMaskProvider(MaskProvider):
    """
    Ground truth mask provider.

    Uses pre-existing GT masks from the dataset.
    This serves as the performance upper bound for CAST.
    """

    def __init__(self, mask_root: Optional[str] = None):
        """
        Args:
            mask_root: Root directory for mask files (optional, can use meta['mask_path'])
        """
        self.mask_root = mask_root

    def get_mask(
        self,
        image: Image.Image,
        question: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Load and return ground truth mask.

        Expects meta to contain either:
        - 'mask_path': Full path to mask file
        - 'mask': Pre-loaded mask array
        - 'image_mask': Pre-loaded PIL mask image
        """
        if meta is None:
            # Return full image mask if no metadata
            return np.ones((image.height, image.width), dtype=np.uint8)

        # Check for pre-loaded mask
        if 'mask' in meta and meta['mask'] is not None:
            mask = meta['mask']
            if isinstance(mask, np.ndarray):
                return (mask > 0).astype(np.uint8)

        if 'image_mask' in meta and meta['image_mask'] is not None:
            mask_img = meta['image_mask']
            if isinstance(mask_img, Image.Image):
                mask = np.array(mask_img)
                return (mask > 0).astype(np.uint8)

        # Try to load from path
        if 'mask_path' in meta and meta['mask_path'] is not None:
            import os
            mask_path = meta['mask_path']
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                mask = np.array(mask_img)
                return (mask > 0).astype(np.uint8)

        # Create mask from bbox if available
        if 'bbox' in meta and meta['bbox'] is not None:
            bbox = meta['bbox']
            if bbox != (0.0, 0.0, 0.0, 0.0):
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(image.width, x1))
                y1 = max(0, min(image.height, y1))
                x2 = max(0, min(image.width, x2))
                y2 = max(0, min(image.height, y2))
                mask[y1:y2, x1:x2] = 1
                return mask

        # Fallback: return full image mask
        return np.ones((image.height, image.width), dtype=np.uint8)


class NullMaskProvider(MaskProvider):
    """
    Null mask provider - returns empty masks.

    This serves as the baseline (no ROI guidance).
    The attention mask will be all zeros, effectively disabling CAST guidance.
    """

    def get_mask(
        self,
        image: Image.Image,
        question: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Return an empty (all zeros) mask.
        """
        return np.zeros((image.height, image.width), dtype=np.uint8)


class AutoMaskProvider(MaskProvider):
    """
    Automatic ROI mask provider.

    Uses proposal generation and counterfactual selection to automatically
    discover question-relevant ROI without ground truth annotations.

    Pipeline:
    1. Generate ROI proposals using MedSAM3
    2. Filter proposals (area, NMS, morphology)
    3. Score proposals using counterfactual likelihood drop
    4. Select best proposal as ROI mask
    """

    def __init__(
        self,
        proposal_generator=None,
        fallback_generator=None,
        roi_selector=None,
        proposal_cache=None,
        K: int = 16,
        use_prefilter: bool = True,
        prefilter_K: int = 50,
        device: str = "cuda"
    ):
        """
        Args:
            proposal_generator: ProposalGenerator instance for generating mask candidates
            fallback_generator: Fallback ProposalGenerator when primary produces nothing
            roi_selector: ROISelector instance for scoring and selecting masks
            proposal_cache: ProposalCache instance for caching proposals
            K: Number of proposals to evaluate
            use_prefilter: Whether to use cheap prefiltering
            prefilter_K: Number of proposals before prefiltering
            device: Device for computation
        """
        self.proposal_generator = proposal_generator
        self.fallback_generator = fallback_generator
        self.roi_selector = roi_selector
        self.proposal_cache = proposal_cache
        self.K = K
        self.use_prefilter = use_prefilter
        self.prefilter_K = prefilter_K
        self.device = device

    def get_mask(
        self,
        image: Image.Image,
        question: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Automatically discover and return the best ROI mask.

        Steps:
        1. Check cache for pre-computed proposals
        2. Generate proposals if not cached
        3. Filter and preselect proposals
        4. Score proposals using counterfactual method
        5. Return best scoring proposal
        """
        image_id = meta.get('image_id', None) if meta else None

        # Step 1: Try to load from cache
        proposals = None
        if self.proposal_cache is not None and image_id is not None:
            proposals = self.proposal_cache.load(image_id)

        # Step 2: Generate proposals if not cached
        if proposals is None:
            if self.proposal_generator is not None:
                proposals = self.proposal_generator.generate(image)
                # Cache the proposals
                if self.proposal_cache is not None and image_id is not None:
                    self.proposal_cache.save(image_id, proposals)

        # Step 2b: Fallback generator if primary produced nothing
        if (proposals is None or len(proposals) == 0) and self.fallback_generator is not None:
            proposals = self.fallback_generator.generate(image)

        # Check if we have any proposals
        if proposals is None or len(proposals) == 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)

        # Step 3: Prefilter if enabled
        if self.use_prefilter and self.roi_selector is not None:
            proposals = self.roi_selector.prefilter(
                image, question, proposals,
                topK=min(self.K, len(proposals))
            )
        else:
            proposals = proposals[:self.K]

        # Step 4: Score proposals
        if self.roi_selector is None:
            # No selector - return first proposal
            return proposals[0] if len(proposals) > 0 else np.zeros((image.height, image.width), dtype=np.uint8)

        # Generate draft answer for scoring
        y_hat = None
        if meta is not None and 'answer' in meta:
            # Use GT answer for closed-set tasks
            y_hat = meta['answer']

        best_mask = None
        best_score = float('-inf')

        for mask in proposals:
            score = self.roi_selector.score(image, question, mask, y_hat)
            if score > best_score:
                best_score = score
                best_mask = mask

        # Step 5: Post-process and return
        if best_mask is not None:
            best_mask = self._postprocess_mask(best_mask)
            return best_mask

        return np.zeros((image.height, image.width), dtype=np.uint8)

    def _postprocess_mask(self, mask: np.ndarray, min_coverage: float = 0.05) -> np.ndarray:
        """
        Post-process the selected mask.

        Steps:
        - Ensure binary values
        - Morphological closing for hole filling
        - Dilate to reach minimum coverage for effective CFG
        - If postprocessing would empty the mask, return original binary mask

        Args:
            mask: Input mask array
            min_coverage: Minimum fraction of image that should be covered (default 5%)
        """
        import cv2

        # Ensure binary
        mask_orig = (mask > 0.5).astype(np.uint8)

        if mask_orig.sum() == 0:
            return mask_orig

        # Morphological closing (fill holes)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_proc = cv2.morphologyEx(mask_orig, cv2.MORPH_CLOSE, kernel_small)

        # Remove tiny noise components (< 0.1% of image area)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_proc, connectivity=8)
        if num_labels > 1:
            min_area = mask_proc.shape[0] * mask_proc.shape[1] * 0.001
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    mask_proc[labels == i] = 0

        # If cleaning emptied the mask, use original
        if mask_proc.sum() == 0:
            mask_proc = mask_orig.copy()

        # Dilate to reach minimum coverage for effective CFG
        total_pixels = mask_proc.shape[0] * mask_proc.shape[1]
        coverage = mask_proc.sum() / total_pixels
        if coverage < min_coverage:
            # Iteratively dilate until reaching min_coverage
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            while coverage < min_coverage:
                mask_proc = cv2.dilate(mask_proc, kernel_dilate, iterations=1)
                coverage = mask_proc.sum() / total_pixels

        return mask_proc
