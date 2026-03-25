"""
Caching utilities for Auto-ARCD.

Caches proposal masks to avoid recomputation during experiments.
"""

import os
import json
import hashlib
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image


class ProposalCache:
    """
    Cache for mask proposals.

    Stores proposals as compressed numpy arrays indexed by image_id.
    """

    def __init__(
        self,
        cache_dir: str,
        prefix: str = "proposals"
    ):
        """
        Args:
            cache_dir: Directory to store cached proposals
            prefix: Prefix for cache files
        """
        self.cache_dir = cache_dir
        self.prefix = prefix
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, image_id: str) -> str:
        """Get cache file path for an image_id."""
        # Sanitize image_id for filesystem
        safe_id = str(image_id).replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{self.prefix}_{safe_id}.npz")

    def save(
        self,
        image_id: str,
        proposals: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save proposals to cache.

        Args:
            image_id: Unique identifier for the image
            proposals: List of binary masks
            metadata: Optional metadata to save alongside
        """
        cache_path = self._get_cache_path(image_id)

        # Stack masks into single array
        if len(proposals) > 0:
            masks = np.stack(proposals, axis=0)
        else:
            masks = np.array([])

        # Save with compression
        save_dict = {'masks': masks}
        if metadata is not None:
            # Convert metadata to JSON string
            save_dict['metadata'] = np.array([json.dumps(metadata)])

        np.savez_compressed(cache_path, **save_dict)

    def load(
        self,
        image_id: str
    ) -> Optional[List[np.ndarray]]:
        """
        Load proposals from cache.

        Args:
            image_id: Unique identifier for the image

        Returns:
            List of binary masks, or None if not cached
        """
        cache_path = self._get_cache_path(image_id)

        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path)
            masks = data['masks']

            if len(masks.shape) == 0 or masks.size == 0:
                return []

            return [masks[i] for i in range(masks.shape[0])]
        except Exception as e:
            print(f"Warning: Failed to load cache for {image_id}: {e}")
            return None

    def load_with_metadata(
        self,
        image_id: str
    ) -> Optional[tuple]:
        """
        Load proposals and metadata from cache.

        Returns:
            Tuple of (proposals, metadata) or None if not cached
        """
        cache_path = self._get_cache_path(image_id)

        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            masks = data['masks']

            if len(masks.shape) == 0 or masks.size == 0:
                proposals = []
            else:
                proposals = [masks[i] for i in range(masks.shape[0])]

            metadata = None
            if 'metadata' in data:
                metadata = json.loads(str(data['metadata'][0]))

            return proposals, metadata
        except Exception as e:
            print(f"Warning: Failed to load cache for {image_id}: {e}")
            return None

    def exists(self, image_id: str) -> bool:
        """Check if proposals are cached for the given image_id."""
        return os.path.exists(self._get_cache_path(image_id))

    def delete(self, image_id: str):
        """Delete cached proposals for the given image_id."""
        cache_path = self._get_cache_path(image_id)
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def clear(self):
        """Clear all cached proposals."""
        import glob
        pattern = os.path.join(self.cache_dir, f"{self.prefix}_*.npz")
        for path in glob.glob(pattern):
            os.remove(path)

    def list_cached(self) -> List[str]:
        """List all cached image_ids."""
        import glob
        pattern = os.path.join(self.cache_dir, f"{self.prefix}_*.npz")
        paths = glob.glob(pattern)
        ids = []
        for path in paths:
            basename = os.path.basename(path)
            # Remove prefix and .npz
            image_id = basename[len(self.prefix) + 1:-4]
            ids.append(image_id)
        return ids


class SelectionCache:
    """
    Cache for ROI selection results.

    Stores selected masks and scores indexed by (image_id, question_hash).
    """

    def __init__(
        self,
        cache_dir: str,
        prefix: str = "selection"
    ):
        """
        Args:
            cache_dir: Directory to store cached selections
            prefix: Prefix for cache files
        """
        self.cache_dir = cache_dir
        self.prefix = prefix
        os.makedirs(cache_dir, exist_ok=True)

        # In-memory index for faster lookups
        self._index_path = os.path.join(cache_dir, f"{prefix}_index.json")
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, str]]:
        """Load cache index from disk."""
        if os.path.exists(self._index_path):
            with open(self._index_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f)

    def _get_question_hash(self, question: str) -> str:
        """Get hash of question for indexing."""
        return hashlib.md5(question.encode()).hexdigest()[:12]

    def _get_cache_path(self, image_id: str, question_hash: str) -> str:
        """Get cache file path."""
        safe_id = str(image_id).replace("/", "_").replace("\\", "_")
        return os.path.join(
            self.cache_dir,
            f"{self.prefix}_{safe_id}_{question_hash}.npz"
        )

    def save(
        self,
        image_id: str,
        question: str,
        selected_mask: np.ndarray,
        score: float,
        all_scores: Optional[List[float]] = None
    ):
        """
        Save selection result to cache.

        Args:
            image_id: Image identifier
            question: Question string
            selected_mask: Selected mask
            score: Score of selected mask
            all_scores: Optional list of all proposal scores
        """
        question_hash = self._get_question_hash(question)
        cache_path = self._get_cache_path(image_id, question_hash)

        save_dict = {
            'mask': selected_mask,
            'score': np.array([score])
        }
        if all_scores is not None:
            save_dict['all_scores'] = np.array(all_scores)

        np.savez_compressed(cache_path, **save_dict)

        # Update index
        if image_id not in self._index:
            self._index[image_id] = {}
        self._index[image_id][question_hash] = {
            'path': cache_path,
            'score': float(score)
        }
        self._save_index()

    def load(
        self,
        image_id: str,
        question: str
    ) -> Optional[tuple]:
        """
        Load selection result from cache.

        Returns:
            Tuple of (mask, score) or None if not cached
        """
        question_hash = self._get_question_hash(question)
        cache_path = self._get_cache_path(image_id, question_hash)

        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path)
            mask = data['mask']
            score = float(data['score'][0])
            return mask, score
        except Exception as e:
            print(f"Warning: Failed to load selection cache: {e}")
            return None

    def exists(self, image_id: str, question: str) -> bool:
        """Check if selection is cached."""
        question_hash = self._get_question_hash(question)
        return (
            image_id in self._index and
            question_hash in self._index[image_id]
        )
