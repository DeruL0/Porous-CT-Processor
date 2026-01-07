"""
Shared segmentation cache for processors.
Stores intermediate results on disk to avoid memory issues and prevent duplicate computation.
"""

import os
import tempfile
import numpy as np
from typing import Optional, Dict, Any
import gc


class SegmentationCache:
    """
    Shared cache for segmentation results between PoreExtractionProcessor and PoreToSphereProcessor.
    Uses disk-backed storage to avoid memory issues with large volumes.
    
    Usage:
        cache = get_segmentation_cache()
        
        # Store result
        cache.store_pores_mask(volume_id, pores_mask, metadata)
        
        # Retrieve result
        result = cache.get_pores_mask(volume_id)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self._prefix = "porous_seg"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._memmap_files: Dict[str, str] = {}
    
    def _get_volume_id(self, raw_data: np.ndarray, threshold: int) -> str:
        """Generate unique ID for a volume + threshold combination."""
        return f"{id(raw_data)}_{threshold}"
    
    def has_cache(self, volume_id: str) -> bool:
        """Check if cache exists for given volume ID."""
        return volume_id in self._cache
    
    def store_pores_mask(self, volume_id: str, pores_mask: np.ndarray, 
                         metadata: Optional[Dict] = None, use_disk: bool = True) -> None:
        """
        Store pores mask in cache.
        
        Args:
            volume_id: Unique identifier for this volume
            pores_mask: Boolean mask of pore voxels
            metadata: Optional metadata (porosity, pore count, etc.)
            use_disk: If True, store on disk as memory-mapped file
        """
        if use_disk and pores_mask.nbytes > 100 * 1024 * 1024:  # > 100MB
            # Store on disk
            filename = os.path.join(self.cache_dir, f"{self._prefix}_{volume_id}.dat")
            mmap = np.memmap(filename, dtype=pores_mask.dtype, mode='w+', shape=pores_mask.shape)
            mmap[:] = pores_mask[:]
            mmap.flush()
            
            self._memmap_files[volume_id] = filename
            self._cache[volume_id] = {
                'shape': pores_mask.shape,
                'dtype': pores_mask.dtype,
                'disk': True,
                'filename': filename,
                'metadata': metadata or {}
            }
            print(f"[SegCache] Stored pores_mask to disk: {filename}")
        else:
            # Store in memory
            self._cache[volume_id] = {
                'pores_mask': pores_mask.copy(),
                'disk': False,
                'metadata': metadata or {}
            }
            print(f"[SegCache] Stored pores_mask in memory: {volume_id}")
    
    def get_pores_mask(self, volume_id: str) -> Optional[np.ndarray]:
        """
        Retrieve pores mask from cache.
        
        Returns:
            pores_mask array or None if not cached
        """
        if volume_id not in self._cache:
            return None
        
        entry = self._cache[volume_id]
        
        if entry.get('disk'):
            # Load from disk as memmap (read-only)
            filename = entry['filename']
            if os.path.exists(filename):
                return np.memmap(filename, dtype=entry['dtype'], mode='r', shape=entry['shape'])
            return None
        else:
            return entry.get('pores_mask')
    
    def get_metadata(self, volume_id: str) -> Optional[Dict]:
        """Get cached metadata for a volume."""
        if volume_id not in self._cache:
            return None
        return self._cache[volume_id].get('metadata')
    
    def clear(self, volume_id: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            volume_id: Specific entry to clear. If None, clears all.
        """
        if volume_id:
            if volume_id in self._cache:
                entry = self._cache[volume_id]
                if entry.get('disk') and 'filename' in entry:
                    try:
                        os.remove(entry['filename'])
                    except:
                        pass
                del self._cache[volume_id]
        else:
            # Clear all
            for vid, entry in list(self._cache.items()):
                if entry.get('disk') and 'filename' in entry:
                    try:
                        os.remove(entry['filename'])
                    except:
                        pass
            self._cache.clear()
            self._memmap_files.clear()
        gc.collect()
        print(f"[SegCache] Cleared cache")
    
    def __del__(self):
        try:
            self.clear()
        except:
            pass


# Global singleton instance
_global_seg_cache: Optional[SegmentationCache] = None


def get_segmentation_cache() -> SegmentationCache:
    """Get the global segmentation cache."""
    global _global_seg_cache
    if _global_seg_cache is None:
        _global_seg_cache = SegmentationCache()
    return _global_seg_cache


def clear_segmentation_cache():
    """Clear the global segmentation cache."""
    global _global_seg_cache
    if _global_seg_cache is not None:
        _global_seg_cache.clear()
    gc.collect()
