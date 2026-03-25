"""
ROI Selection: Select the best ROI mask from proposals.

This module implements the counterfactual likelihood drop method
for selecting question-relevant ROI masks.

Core idea: If a region is key evidence for answering a question,
occluding that region should significantly reduce the model's
confidence in the answer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2


class ROISelector(ABC):
    """
    Abstract base class for ROI selection.
    """

    @abstractmethod
    def score(
        self,
        image: Image.Image,
        question: str,
        mask: np.ndarray,
        answer: Optional[str] = None
    ) -> float:
        """
        Score a mask for its relevance to the question.

        Args:
            image: PIL Image (RGB)
            question: Question string
            mask: Binary mask to score
            answer: Optional answer for teacher forcing

        Returns:
            Score (higher = more relevant)
        """
        pass

    def prefilter(
        self,
        image: Image.Image,
        question: str,
        proposals: List[np.ndarray],
        topK: int = 16
    ) -> List[np.ndarray]:
        """
        Cheap prefiltering of proposals.

        Default implementation: just return first K proposals.
        Override for smarter prefiltering.
        """
        return proposals[:topK]


class CounterfactualSelector(ROISelector):
    """
    Select ROI using counterfactual likelihood drop.

    For each candidate mask:
    1. Generate occluded image (blur/mean the masked region)
    2. Compute log-likelihood of answer with original image
    3. Compute log-likelihood of answer with occluded image
    4. Score = ll_original - ll_occluded (higher = more relevant)

    Intuition: If masking a region significantly reduces answer confidence,
    that region must be important for answering the question.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        occlusion_modes: List[str] = ["blur", "mean"],
        lambda_area: float = 0.1,
        mu_consistency: float = 0.1,
        T_score: int = 32,
        T_draft: int = 32,
        device: str = "cuda"
    ):
        """
        Args:
            model: VLM model for computing log-likelihood
            processor: Processor for the model
            occlusion_modes: List of occlusion methods to use
            lambda_area: Area penalty coefficient
            mu_consistency: Consistency bonus coefficient
            T_score: Number of tokens to use for scoring
            T_draft: Number of tokens to generate for draft answer
            device: Device for computation
        """
        self.model = model
        self.processor = processor
        self.occlusion_modes = occlusion_modes
        self.lambda_area = lambda_area
        self.mu_consistency = mu_consistency
        self.T_score = T_score
        self.T_draft = T_draft
        self.device = device

    def _process_inputs(self, text, images):
        """
        Call the custom processor and assemble input_ids from prompt_chunks + image_ids_pad.

        The custom Phi-3.5V processor doesn't return input_ids directly when images
        are present. Instead it returns prompt_chunks and image_ids_pad that need
        to be assembled.
        """
        raw = self.processor(text=text, images=images, return_tensors='pt')

        if 'input_ids' in raw:
            # Standard processor path
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in raw.items()}

        # Custom processor: assemble input_ids from prompt_chunks + image_ids_pad
        prompt_chunks = raw['prompt_chunks']
        image_ids_pad = raw['image_ids_pad']

        # Interleave prompt_chunks and image_ids_pad (same logic as insert_separator)
        sep_list = list(image_ids_pad)
        if len(prompt_chunks) > len(sep_list):
            sep_list.append([])
        merged = [ele for sublist in zip(prompt_chunks, sep_list) for ele in sublist]

        input_ids_list = []
        for tokens in merged:
            input_ids_list.extend(tokens)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)

        result = {
            'input_ids': input_ids,
            'pixel_values': raw['pixel_values'].to(self.device),
            'image_sizes': raw['image_sizes'].to(self.device),
        }
        return result

    def occlude(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mode: str = "blur"
    ) -> np.ndarray:
        """
        Apply occlusion to the masked region.

        Args:
            image: Input image (H, W, 3) as numpy array
            mask: Binary mask (H, W), 1 = ROI to occlude
            mode: Occlusion mode:
                - "zero": Set ROI to black
                - "mean": Set ROI to image mean
                - "blur": Gaussian blur the ROI (recommended)
                - "noise": Replace ROI with random noise

        Returns:
            Occluded image (H, W, 3)
        """
        occluded = image.copy()
        mask_bool = mask.astype(bool)

        if mode == "zero":
            occluded[mask_bool] = 0

        elif mode == "mean":
            mean_color = image.mean(axis=(0, 1))
            occluded[mask_bool] = mean_color

        elif mode == "blur":
            # Heavy Gaussian blur
            blurred = cv2.GaussianBlur(image, (51, 51), 30)
            occluded[mask_bool] = blurred[mask_bool]

        elif mode == "noise":
            noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
            occluded[mask_bool] = noise[mask_bool]

        else:
            raise ValueError(f"Unknown occlusion mode: {mode}")

        return occluded

    def draft_answer(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = None
    ) -> str:
        """
        Generate a draft answer using greedy decoding.

        Args:
            image: PIL Image
            question: Question string
            max_new_tokens: Max tokens to generate

        Returns:
            Draft answer string
        """
        if self.model is None or self.processor is None:
            return ""

        max_new_tokens = max_new_tokens or self.T_draft

        # Prepare prompt
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = self._process_inputs(text=prompt, images=[image])

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                image_sizes=inputs['image_sizes'],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                num_beams=1,
            )

        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True,
        )[0].strip()

        return generated_text

    @torch.no_grad()
    def loglikelihood(
        self,
        image: Image.Image,
        question: str,
        answer: str,
        T: int = None
    ) -> float:
        """
        Compute teacher-forcing log-likelihood of answer.

        Args:
            image: PIL Image
            question: Question string
            answer: Answer string to compute likelihood for
            T: Number of tokens to use (None = all tokens)

        Returns:
            Sum of log probabilities: sum(log p(y_t | y_<t, image, question))
        """
        if self.model is None or self.processor is None:
            return 0.0

        T = T or self.T_score

        # Prepare prompt with answer
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        full_prompt = prompt + answer

        inputs = self._process_inputs(text=full_prompt, images=[image])

        # Get the length of the prompt (without answer)
        # Compute prompt_length by subtracting answer token count from total
        answer_tokens = self.processor.tokenizer(answer, add_special_tokens=False, return_tensors='pt')
        prompt_length = inputs['input_ids'].size(1) - answer_tokens['input_ids'].size(1)

        # Forward pass
        outputs = self.model(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            image_sizes=inputs['image_sizes'],
        )

        logits = outputs.logits
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, prompt_length - 1:-1, :]
        shift_labels = inputs['input_ids'][:, prompt_length:]

        # Limit to T tokens
        if shift_labels.size(1) > T:
            shift_logits = shift_logits[:, :T, :]
            shift_labels = shift_labels[:, :T]

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probabilities
        total_ll = token_log_probs.sum().item()

        return total_ll

    def score(
        self,
        image: Image.Image,
        question: str,
        mask: np.ndarray,
        answer: Optional[str] = None
    ) -> float:
        """
        Score a mask using counterfactual likelihood drop.

        Higher score = more relevant to the question.
        """
        if self.model is None:
            # Fallback: score by mask area (prefer medium-sized masks)
            area = mask.mean()
            return -abs(area - 0.2)  # Prefer ~20% coverage

        # Generate draft answer if not provided
        if answer is None or answer == "":
            answer = self.draft_answer(image, question)

        if answer == "":
            return 0.0

        img_np = np.array(image)

        # Compute baseline likelihood
        ll_full = self.loglikelihood(image, question, answer)

        # Compute likelihood with different occlusion modes
        deltas = []
        for mode in self.occlusion_modes:
            # Create occluded image
            img_occ = self.occlude(img_np, mask, mode=mode)
            img_occ_pil = Image.fromarray(img_occ)

            # Compute likelihood with occluded image
            ll_occ = self.loglikelihood(img_occ_pil, question, answer)

            # Likelihood drop (higher = more important region)
            delta = ll_full - ll_occ
            deltas.append(delta)

        # Average delta across modes
        mean_delta = np.mean(deltas)

        # Area penalty (avoid selecting full image)
        area_penalty = self.lambda_area * mask.mean()

        # Consistency bonus (reward if all modes agree)
        if len(deltas) > 1:
            consistency = -np.var(deltas)
        else:
            consistency = 0.0

        # Final score
        score = mean_delta - area_penalty + self.mu_consistency * consistency

        return score

    def prefilter(
        self,
        image: Image.Image,
        question: str,
        proposals: List[np.ndarray],
        topK: int = 16
    ) -> List[np.ndarray]:
        """
        Cheap prefiltering using area-based scoring.

        Prefer medium-sized masks that are not too small or too large.
        """
        if len(proposals) <= topK:
            return proposals

        # Score by area preference (prefer 10-40% coverage)
        scores = []
        for mask in proposals:
            area = mask.mean()
            # Gaussian preference around 25%
            area_score = np.exp(-((area - 0.25) ** 2) / (2 * 0.15 ** 2))
            scores.append(area_score)

        # Sort by score
        indices = np.argsort(scores)[::-1][:topK]
        return [proposals[i] for i in indices]


class RandomSelector(ROISelector):
    """
    Random ROI selection baseline.

    Selects a random proposal from the filtered list.
    Used as ablation baseline to verify that counterfactual scoring
    provides actual signal over random selection.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def score(
        self,
        image: Image.Image,
        question: str,
        mask: np.ndarray,
        answer: Optional[str] = None
    ) -> float:
        return self.rng.random()

    def prefilter(
        self,
        image: Image.Image,
        question: str,
        proposals: List[np.ndarray],
        topK: int = 16
    ) -> List[np.ndarray]:
        n = min(topK, len(proposals))
        indices = self.rng.choice(len(proposals), n, replace=False)
        return [proposals[i] for i in indices]


class HeatmapSelector(ROISelector):
    """
    Select ROI using attention heatmap overlap.

    Uses VLM attention patterns to identify question-relevant regions.
    Scores masks by overlap with attention heatmap.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        device: str = "cuda"
    ):
        self.model = model
        self.processor = processor
        self.device = device

    def get_attention_heatmap(
        self,
        image: Image.Image,
        question: str
    ) -> np.ndarray:
        """
        Extract attention heatmap from VLM.

        Returns a heatmap of shape (H, W) showing attention weights.
        """
        # This would require model-specific implementation
        # to extract cross-attention between text and image tokens
        # For now, return uniform heatmap
        return np.ones((image.height, image.width))

    def score(
        self,
        image: Image.Image,
        question: str,
        mask: np.ndarray,
        answer: Optional[str] = None
    ) -> float:
        """
        Score mask by overlap with attention heatmap.
        """
        heatmap = self.get_attention_heatmap(image, question)

        # Resize heatmap to match mask size if needed
        if heatmap.shape != mask.shape:
            heatmap = cv2.resize(
                heatmap,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Compute overlap score
        overlap = (mask * heatmap).sum() / (mask.sum() + 1e-8)

        return overlap
