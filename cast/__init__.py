# CAST: Contrastive Anatomical Spatial-Temporal Decoding
# Annotation-free anatomical region-guided contrastive decoding for Med-VLMs

from .mask_provider import (
    MaskProvider,
    GTMaskProvider,
    NullMaskProvider,
    AutoMaskProvider,
)
from .roi_selection import (
    ROISelector,
    CounterfactualSelector,
    RandomSelector,
)
from .proposal_generator import (
    ProposalGenerator,
    MedSAM3ProposalGenerator,
    SimpleProposalGenerator,
)
from .cache import ProposalCache
from .guidance import ProbCFGLogitsProcessor
from .utils import txt_highlight_mask

__all__ = [
    "MaskProvider",
    "GTMaskProvider",
    "NullMaskProvider",
    "AutoMaskProvider",
    "ROISelector",
    "CounterfactualSelector",
    "RandomSelector",
    "ProposalGenerator",
    "MedSAM3ProposalGenerator",
    "SimpleProposalGenerator",
    "ProposalCache",
    "ProbCFGLogitsProcessor",
    "txt_highlight_mask",
]
