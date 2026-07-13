"""
Caching utilities for CAST.

Caches proposal masks to avoid recomputation during experiments.
"""

import os
import json
from typing import List, Optional, Dict, Any
import numpy as np


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
