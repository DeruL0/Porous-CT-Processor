"""
Disk cache manager for large volume data.
Provides memory-mapped arrays for out-of-core processing of large datasets.
"""

import os
import tempfile
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gc

logger = logging.getLogger(__name__)


class DiskCacheManager:
    """
    Manages disk-based caching for large numpy arrays.
    Uses memory-mapped files to avoid RAM limitations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, prefix: str = "porous_cache"):
        """
        Args:
            cache_dir: Directory for cache files. Uses system temp if None.
            prefix: Prefix for cache file names.
        """
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.prefix = prefix
        self._cache_files: Dict[str, str] = {}
        self._cache_arrays: Dict[str, np.memmap] = {}
    
    def create_array(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> np.memmap:
        """
        Create a memory-mapped array on disk.
        
        Args:
            name: Unique identifier for this array.
            shape: Shape of the array.
            dtype: Data type.
            
        Returns:
            Memory-mapped numpy array.
        """
        # Clean up existing array with same name
        self.release(name)
        
        # Create memory-mapped file
        filename = os.path.join(self.cache_dir, f"{self.prefix}_{name}_{id(self)}.dat")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        
        self._cache_files[name] = filename
        self._cache_arrays[name] = arr
        
        print(f"[DiskCache] Created '{name}': {shape}, {dtype}, {self._format_size(arr.nbytes)}")
        return arr
    
    def get_array(self, name: str) -> Optional[np.memmap]:
        """Get a cached array by name."""
        return self._cache_arrays.get(name)
    
    def release(self, name: str):
        """Release a specific cached array and delete its file."""
        if name in self._cache_arrays:
            del self._cache_arrays[name]
            gc.collect()
        
        if name in self._cache_files:
            try:
                os.remove(self._cache_files[name])
            except OSError as e:
                logger.warning(f"Failed to remove cache file {self._cache_files[name]}: {e}")
            del self._cache_files[name]
    
    def release_all(self):
        """Release all cached arrays and delete all cache files."""
        names = list(self._cache_files.keys())
        for name in names:
            self.release(name)
        gc.collect()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def __del__(self):
        """Clean up on deletion."""
        try:
            self.release_all()
        except Exception as e:
            # Log but don't raise in destructor - may be called during interpreter shutdown
            logger.debug(f"Cache cleanup during destruction failed: {e}")


class ChunkedProcessor:
    """
    Processes large volumes in chunks to avoid memory exhaustion.
    """
    
    def __init__(self, chunk_size: int = 64):
        """
        Args:
            chunk_size: Number of slices to process at a time.
        """
        self.chunk_size = chunk_size
        self.cache = DiskCacheManager()
    
    def process_chunked(self, volume: np.ndarray, func, 
                        output_dtype=np.float32,
                        callback=None) -> np.memmap:
        """
        Process a volume in chunks, storing results on disk.
        
        Args:
            volume: Input volume array.
            func: Function to apply to each chunk. Signature: func(chunk) -> result.
            output_dtype: Data type for output array.
            callback: Optional progress callback.
            
        Returns:
            Memory-mapped result array.
        """
        shape = volume.shape
        n_slices = shape[0]
        n_chunks = (n_slices + self.chunk_size - 1) // self.chunk_size
        
        # Create output array on disk
        output = self.cache.create_array("output", shape, dtype=output_dtype)
        
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, n_slices)
            
            # Process chunk
            chunk = volume[start:end]
            result = func(chunk)
            output[start:end] = result
            
            # Flush to disk
            output.flush()
            
            if callback:
                progress = int(100 * (i + 1) / n_chunks)
                callback(progress, f"Processing chunk {i+1}/{n_chunks}...")
            
            # Free chunk memory
            del chunk, result
            gc.collect()
        
        return output
    
    def cleanup(self):
        """Release all cached data."""
        self.cache.release_all()


# Global cache instance for shared use
_global_cache: Optional[DiskCacheManager] = None


def get_disk_cache() -> DiskCacheManager:
    """Get the global disk cache manager."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DiskCacheManager()
    return _global_cache


def clear_disk_cache():
    """Clear the global disk cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.release_all()
        _global_cache = None
    gc.collect()


# ==========================================
# Segmentation Cache (for PNM processing)
# ==========================================

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
                    except OSError as e:
                        logger.warning(f"Failed to remove segmentation cache file: {e}")
                del self._cache[volume_id]
        else:
            # Clear all
            for vid, entry in list(self._cache.items()):
                if entry.get('disk') and 'filename' in entry:
                    try:
                        os.remove(entry['filename'])
                    except OSError as e:
                        logger.warning(f"Failed to remove segmentation cache file {entry['filename']}: {e}")
            self._cache.clear()
            self._memmap_files.clear()
        gc.collect()
        print(f"[SegCache] Cleared cache")
    
    def __del__(self):
        try:
            self.clear()
        except Exception as e:
            # Log but don't raise in destructor
            logger.debug(f"SegmentationCache cleanup during destruction failed: {e}")


# Global singleton instance for segmentation cache
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


# ==========================================
# Time Series PNM Cache (for 4DCT tracking)
# ==========================================

class TimeSeriesPNMCache:
    """
    Cache for 4DCT time series PNM tracking results.
    Stores TimeSeriesPNM objects on disk using pickle serialization
    to avoid regenerating PNM when switching between views.
    
    Usage:
        cache = get_timeseries_pnm_cache()
        
        # Store time series PNM after tracking
        cache_key = cache.generate_key(volumes, threshold)
        cache.store(cache_key, time_series_pnm, reference_mesh)
        
        # Retrieve when switching back to PNM view
        cached_data = cache.get(cache_key)
        if cached_data:
            time_series_pnm, reference_mesh = cached_data
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self._prefix = "porous_4dct_pnm"
        self._cache: Dict[str, Dict[str, Any]] = {}
        
    def generate_key(self, volumes: List, threshold: int) -> str:
        """
        Generate unique cache key from volumes and threshold.
        Uses volume IDs and folder names to create a stable identifier.
        
        Args:
            volumes: List of VolumeData objects
            threshold: HU threshold value
            
        Returns:
            Unique cache key string
        """
        # Create key from volume IDs and folder names
        volume_ids = []
        for vol in volumes:
            folder_name = vol.metadata.get('folder_name', '')
            vol_id = id(vol.raw_data) if vol.raw_data is not None else 0
            volume_ids.append(f"{folder_name}_{vol_id}")
        
        key_str = "_".join(volume_ids) + f"_t{threshold}"
        # Use hash to keep key length manageable
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def has_cache(self, cache_key: str) -> bool:
        """Check if cache exists for given key."""
        return cache_key in self._cache
    
    def store(self, cache_key: str, time_series_pnm, reference_mesh, 
              reference_snapshot) -> None:
        """
        Store time series PNM data to cache.
        
        Args:
            cache_key: Unique cache identifier
            time_series_pnm: TimeSeriesPNM object with tracking results
            reference_mesh: Reference mesh (VolumeData) from t=0
            reference_snapshot: Reference PNMSnapshot from t=0
        """
        import pickle
        
        cache_data = {
            'time_series_pnm': time_series_pnm,
            'reference_mesh': reference_mesh,
            'reference_snapshot': reference_snapshot
        }
        
        filename = os.path.join(self.cache_dir, f"{self._prefix}_{cache_key}.pkl")
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._cache[cache_key] = {
                'filename': filename,
                'num_timepoints': time_series_pnm.num_timepoints,
                'num_pores': time_series_pnm.num_reference_pores
            }
            
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"[4DCT PNM Cache] Stored: {cache_key[:8]}... "
                  f"({time_series_pnm.num_timepoints} timepoints, "
                  f"{time_series_pnm.num_reference_pores} pores, "
                  f"{size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to store 4DCT PNM cache: {e}")
    
    def get(self, cache_key: str) -> Optional[Tuple]:
        """
        Retrieve time series PNM data from cache.
        
        Returns:
            Tuple of (time_series_pnm, reference_mesh, reference_snapshot) or None
        """
        if cache_key not in self._cache:
            return None
        
        import pickle
        
        entry = self._cache[cache_key]
        filename = entry['filename']
        
        if not os.path.exists(filename):
            # Cache file was deleted, remove entry
            del self._cache[cache_key]
            return None
        
        try:
            with open(filename, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"[4DCT PNM Cache] Loaded: {cache_key[:8]}... "
                  f"({entry['num_timepoints']} timepoints, "
                  f"{entry['num_pores']} pores)")
            
            return (
                cache_data['time_series_pnm'],
                cache_data['reference_mesh'],
                cache_data['reference_snapshot']
            )
        except Exception as e:
            logger.error(f"Failed to load 4DCT PNM cache: {e}")
            return None
    
    def clear(self, cache_key: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            cache_key: Specific entry to clear. If None, clears all.
        """
        if cache_key:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                filename = entry['filename']
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except OSError as e:
                    logger.warning(f"Failed to remove 4DCT PNM cache file: {e}")
                del self._cache[cache_key]
        else:
            # Clear all
            for key, entry in list(self._cache.items()):
                filename = entry['filename']
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except OSError as e:
                    logger.warning(f"Failed to remove 4DCT PNM cache file {filename}: {e}")
            self._cache.clear()
            print(f"[4DCT PNM Cache] Cleared all cache")
        
        gc.collect()
    
    def __del__(self):
        try:
            self.clear()
        except Exception as e:
            logger.debug(f"TimeSeriesPNMCache cleanup during destruction failed: {e}")


# Global singleton instance for time series PNM cache
_global_ts_pnm_cache: Optional[TimeSeriesPNMCache] = None


def get_timeseries_pnm_cache() -> TimeSeriesPNMCache:
    """Get the global time series PNM cache."""
    global _global_ts_pnm_cache
    if _global_ts_pnm_cache is None:
        _global_ts_pnm_cache = TimeSeriesPNMCache()
    return _global_ts_pnm_cache


def clear_timeseries_pnm_cache():
    """Clear the global time series PNM cache."""
    global _global_ts_pnm_cache
    if _global_ts_pnm_cache is not None:
        _global_ts_pnm_cache.clear()
    gc.collect()

