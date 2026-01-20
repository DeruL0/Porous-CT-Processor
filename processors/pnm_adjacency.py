"""
Adjacency detection algorithms for Pore Network Modeling.
GPU-accelerated with CuPy, with CPU fallback.
"""

import time
import numpy as np
from typing import Set, Tuple

from config import GPU_ENABLED
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def find_adjacency(labels_volume: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Detect adjacent pores in a labeled volume.
    
    Uses GPU acceleration when available, falls back to CPU otherwise.
    
    Args:
        labels_volume: 3D integer array where each pore has a unique label (0 = background)
        
    Returns:
        Set of (label_a, label_b) tuples where label_a < label_b
    """
    backend = get_gpu_backend()
    
    if GPU_ENABLED and backend.available and CUPY_AVAILABLE:
        try:
            return _find_adjacency_gpu(labels_volume)
        except Exception as e:
            print(f"[GPU] Adjacency detection failed: {e}, using CPU")
            backend.clear_memory()
    
    return _find_adjacency_cpu(labels_volume)


def _find_adjacency_gpu(labels_volume: np.ndarray) -> Set[Tuple[int, int]]:
    """GPU implementation of adjacency detection using CuPy."""
    import cupy as cp
    
    backend = get_gpu_backend()
    start = time.time()
    
    labels_gpu = cp.asarray(labels_volume)
    adjacency = set()
    
    # Process each axis
    for axis in range(3):
        if axis == 0:
            curr = labels_gpu[:-1, :, :]
            next_ = labels_gpu[1:, :, :]
        elif axis == 1:
            curr = labels_gpu[:, :-1, :]
            next_ = labels_gpu[:, 1:, :]
        else:
            curr = labels_gpu[:, :, :-1]
            next_ = labels_gpu[:, :, 1:]
        
        # Vectorized mask
        mask = (curr > 0) & (next_ > 0) & (curr != next_)
        
        # Explicit GPU->CPU sync with .item()
        has_adjacency = bool(cp.any(mask).item())
        if has_adjacency:
            pairs_a = curr[mask]
            pairs_b = next_[mask]
            
            # Transfer pairs to CPU
            pairs_a_cpu = cp.asnumpy(pairs_a)
            pairs_b_cpu = cp.asnumpy(pairs_b)
            
            # Use numpy unique to reduce duplicates before set operations
            stacked = np.stack([np.minimum(pairs_a_cpu, pairs_b_cpu),
                               np.maximum(pairs_a_cpu, pairs_b_cpu)], axis=1)
            unique_pairs = np.unique(stacked, axis=0)
            
            for a, b in unique_pairs:
                adjacency.add((int(a), int(b)))
            
            del pairs_a, pairs_b
    
    del labels_gpu
    backend.clear_memory()
    
    elapsed = time.time() - start
    print(f"[GPU] Adjacency detection: {elapsed:.2f}s, found {len(adjacency)} connections")
    
    return adjacency


def _find_adjacency_cpu(labels_volume: np.ndarray) -> Set[Tuple[int, int]]:
    """CPU implementation of adjacency detection."""
    start = time.time()
    adjacency = set()
    shape = labels_volume.shape
    chunk_size = 64
    
    def process_axis(axis: int):
        """Process adjacency along a single axis."""
        for i in range(0, shape[axis] - 1, chunk_size):
            end = min(i + chunk_size, shape[axis] - 1)
            slices_curr = [slice(None)] * 3
            slices_next = [slice(None)] * 3
            slices_curr[axis] = slice(i, end)
            slices_next[axis] = slice(i + 1, end + 1)
            
            val_curr = labels_volume[tuple(slices_curr)]
            val_next = labels_volume[tuple(slices_next)]
            mask = (val_curr > 0) & (val_next > 0) & (val_curr != val_next)
            
            if np.any(mask):
                stacked = np.stack([
                    np.minimum(val_curr[mask], val_next[mask]),
                    np.maximum(val_curr[mask], val_next[mask])
                ], axis=1)
                for a, b in np.unique(stacked, axis=0):
                    adjacency.add((int(a), int(b)))
    
    for axis in range(3):
        process_axis(axis)
    
    print(f"[CPU] Adjacency detection: {time.time() - start:.2f}s, found {len(adjacency)} connections")
    return adjacency
