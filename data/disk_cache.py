"""
Disk cache manager for large volume data.
Provides memory-mapped arrays for out-of-core processing of large datasets.
"""

import os
import tempfile
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gc


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
            except:
                pass
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
        except:
            pass


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
