"""
Shared utility functions for processors.
Provides GPU-accelerated operations with CPU fallback.
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import Optional
import time

from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def binary_fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask with GPU acceleration.
    
    Args:
        binary_mask: Binary 3D array (True = foreground)
        
    Returns:
        Filled binary mask
    """
    if not GPU_ENABLED:
        return ndimage.binary_fill_holes(binary_mask)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and backend.get_free_memory_mb() > size_mb * 4:
        try:
            import cupyx.scipy.ndimage as gpu_ndimage
            
            start = time.time()
            mask_gpu = backend.to_gpu(binary_mask)
            filled_gpu = gpu_ndimage.binary_fill_holes(mask_gpu)
            result = backend.to_cpu(filled_gpu)
            
            # Clean up
            del mask_gpu, filled_gpu
            backend.clear_memory()
            
            print(f"[GPU] binary_fill_holes: {time.time() - start:.2f}s")
            return result
        except Exception as e:
            print(f"[GPU] binary_fill_holes failed: {e}, using CPU")
            backend.clear_memory()
    
    # Fallback to CPU
    start = time.time()
    result = ndimage.binary_fill_holes(binary_mask)
    print(f"[CPU] binary_fill_holes: {time.time() - start:.2f}s")
    return result


def distance_transform_edt(binary_mask: np.ndarray, 
                           sampling: Optional[tuple] = None) -> np.ndarray:
    """
    Euclidean Distance Transform with GPU acceleration.
    
    Args:
        binary_mask: Binary 3D array (True = foreground)
        sampling: Voxel spacing (optional)
        
    Returns:
        Distance transform as float32 array
    """
    if not GPU_ENABLED:
        return ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    
    # EDT needs dense float32 output + intermediate arrays
    required_mb = size_mb * 32  # Conservative estimate
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb:
        try:
            import cupyx.scipy.ndimage as gpu_ndimage
            
            start = time.time()
            # Convert to uint8 for CuPy compatibility if boolean
            mask_gpu = backend.to_gpu(binary_mask.astype(np.uint8))
            
            # CuPy EDT
            result_gpu = gpu_ndimage.distance_transform_edt(mask_gpu, sampling=sampling)
            result = backend.to_cpu(result_gpu).astype(np.float32)
            
            del mask_gpu, result_gpu
            backend.clear_memory()
            
            print(f"[GPU] EDT: {time.time() - start:.2f}s")
            return result
        except Exception as e:
            print(f"[GPU] EDT failed: {e}, using CPU")
            backend.clear_memory()
    else:
        if backend.available and size_mb >= GPU_MIN_SIZE_MB:
            print(f"[GPU] EDT skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB free")

    # Fallback
    start = time.time()
    result = ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    print(f"[CPU] EDT: {time.time() - start:.2f}s")
    return result


def find_local_maxima(image: np.ndarray,
                      min_distance: int = 1,
                      labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Find local maxima with GPU acceleration.
    
    This implementation matches skimage.feature.peak_local_max behavior by
    including non-maximum suppression.
    
    Args:
        image: Input image
        min_distance: Minimum distance between peaks
        labels: Optional label image to restrict search
        
    Returns:
        Array of peak coordinates (N, ndim)
    """
    from skimage.feature import peak_local_max
    
    if not GPU_ENABLED:
        return peak_local_max(image, min_distance=min_distance, labels=labels)
    
    backend = get_gpu_backend()
    size_mb = image.nbytes / (1024 * 1024)
    
    # Need ~4x memory: image + max_filtered + is_peak + sorted arrays
    required_mb = size_mb * 4
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8 and CUPY_AVAILABLE:
        try:
            result = _find_local_maxima_gpu_impl(image, min_distance, labels)
            return result
        except Exception as e:
            print(f"[GPU] find_local_maxima failed: {e}, using CPU")
            backend.clear_memory()
    else:
        if backend.available and size_mb >= GPU_MIN_SIZE_MB:
            print(f"[GPU] find_local_maxima skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB free")
    
    # CPU fallback
    start = time.time()
    result = peak_local_max(image, min_distance=min_distance, labels=labels)
    elapsed = time.time() - start
    print(f"[CPU] find_local_maxima: {elapsed:.2f}s, found {len(result)} peaks")
    return result


def _find_local_maxima_gpu_impl(image: np.ndarray,
                                 min_distance: int,
                                 labels: Optional[np.ndarray]) -> np.ndarray:
    """
    GPU implementation of local maxima detection with non-maximum suppression.
    
    Algorithm:
    1. Use maximum_filter to find local maxima candidates
    2. Sort candidates by image value (descending)
    3. Transfer to CPU for NMS (iterative)
    
    This matches the behavior of skimage.feature.peak_local_max.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    
    backend = get_gpu_backend()
    start = time.time()
    
    # Transfer to GPU
    image_gpu = cp.asarray(image.astype(np.float32))
    
    # Step 1: Find local maxima candidates using maximum_filter
    size = 2 * min_distance + 1
    max_filtered = gpu_ndimage.maximum_filter(image_gpu, size=size)
    
    # Initial peak candidates: where image equals local max and is > 0
    is_peak = (image_gpu == max_filtered) & (image_gpu > 0)
    
    # Apply labels mask if provided
    if labels is not None:
        labels_gpu = cp.asarray(labels)
        is_peak = is_peak & (labels_gpu > 0)
        del labels_gpu
    
    # Get candidate coordinates and their values
    candidates = cp.argwhere(is_peak)
    
    if len(candidates) == 0:
        del image_gpu, max_filtered, is_peak
        backend.clear_memory()
        return np.array([]).reshape(0, image.ndim)
    
    # Get values at candidate positions
    candidate_values = image_gpu[is_peak]
    
    del max_filtered, is_peak
    
    # Step 2: Sort by value descending (highest peaks first)
    sort_idx = cp.argsort(-candidate_values)
    sorted_candidates = candidates[sort_idx]
    sorted_values = candidate_values[sort_idx]
    
    del candidates, candidate_values, sort_idx
    
    # Transfer to CPU for NMS
    sorted_candidates_cpu = backend.to_cpu(sorted_candidates)
    sorted_values_cpu = backend.to_cpu(sorted_values)
    
    del sorted_candidates, sorted_values, image_gpu
    backend.clear_memory()
    
    # Step 3: Optimized NMS using KDTree for O(n log n) neighbor queries
    n_candidates = len(sorted_candidates_cpu)
    
    if n_candidates > 10000:
        # Use KDTree for large candidate sets
        from scipy.spatial import cKDTree
        
        tree = cKDTree(sorted_candidates_cpu)
        suppressed = np.zeros(n_candidates, dtype=bool)
        selected_peaks = []
        
        for i in range(n_candidates):
            if suppressed[i]:
                continue
            
            peak = sorted_candidates_cpu[i]
            selected_peaks.append(peak)
            
            # Query neighbors within min_distance
            neighbors = tree.query_ball_point(peak, r=min_distance, p=2)
            
            # Suppress all neighbors (except self, which is already selected)
            for j in neighbors:
                if j > i:  # Only suppress lower-priority (smaller value) candidates
                    suppressed[j] = True
    else:
        # Original vectorized approach for smaller sets
        selected_peaks = []
        suppressed = np.zeros(n_candidates, dtype=bool)
        min_dist_sq = min_distance * min_distance
        
        for i in range(n_candidates):
            if suppressed[i]:
                continue
            
            peak = sorted_candidates_cpu[i]
            selected_peaks.append(peak)
            
            # Vectorized distance calculation for remaining candidates
            if i + 1 < n_candidates:
                remaining = sorted_candidates_cpu[i+1:]
                diff = remaining - peak
                dist_sq = np.sum(diff * diff, axis=1)
                close_mask = dist_sq < min_dist_sq
                suppressed[i+1:][close_mask] = True
    
    result = np.array(selected_peaks) if selected_peaks else np.array([]).reshape(0, image.ndim)
    
    elapsed = time.time() - start
    print(f"[GPU] find_local_maxima: {elapsed:.2f}s, found {len(result)} peaks (NMS applied)")
    
    return result


def watershed_gpu(image: np.ndarray,
                  markers: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  max_iterations: int = 500) -> np.ndarray:
    """
    GPU-accelerated watershed segmentation using iterative label propagation.
    
    This implementation uses a priority-flood approach with CuPy, which works
    on Windows without requiring cucim.
    
    Args:
        image: Input image (typically negative distance transform)
        markers: Initial marker labels (int32, 0 = unlabeled)
        mask: Optional binary mask (True = valid region)
        max_iterations: Maximum iterations for convergence
        
    Returns:
        Labeled segmentation array (int32)
    """
    from skimage.segmentation import watershed as cpu_watershed
    
    if not GPU_ENABLED:
        return cpu_watershed(image, markers, mask=mask)
    
    backend = get_gpu_backend()
    size_mb = image.nbytes / (1024 * 1024)
    
    # Watershed needs ~5x memory: image + labels + neighbors + temp arrays
    required_mb = size_mb * 5
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8 and CUPY_AVAILABLE:
        try:
            result = _watershed_gpu_impl(image, markers, mask, max_iterations)
            return result
        except Exception as e:
            print(f"[GPU] watershed failed: {e}, using CPU")
            backend.clear_memory()
    else:
        if backend.available and size_mb >= GPU_MIN_SIZE_MB:
            print(f"[GPU] watershed skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB free")
    
    # CPU fallback
    start = time.time()
    result = cpu_watershed(image, markers, mask=mask)
    elapsed = time.time() - start
    print(f"[CPU] watershed: {elapsed:.2f}s")
    return result


def _watershed_gpu_impl(image: np.ndarray,
                        markers: np.ndarray,
                        mask: Optional[np.ndarray],
                        max_iterations: int) -> np.ndarray:
    """
    GPU implementation of watershed using priority-based label propagation.
    
    Algorithm:
    1. Initialize labels from markers
    2. Track the "priority" (image value) at which each pixel was labeled
    3. For each unlabeled pixel adjacent to a labeled pixel:
       - Assign the label of the neighbor with the smallest image value
       - Record the priority as the neighbor's image value
    4. Repeat until no changes or max_iterations
    
    This more closely matches CPU watershed by respecting image value ordering.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    
    backend = get_gpu_backend()
    start = time.time()
    
    # Transfer to GPU
    image_gpu = cp.asarray(image.astype(np.float32))
    labels_gpu = cp.asarray(markers.astype(np.int32))
    
    if mask is not None:
        mask_gpu = cp.asarray(mask.astype(np.bool_))
    else:
        mask_gpu = cp.ones(image.shape, dtype=cp.bool_)
    
    # Track the priority (image value) at which each pixel was assigned
    # Initialize with image values for marker pixels, inf for unlabeled
    priority_gpu = cp.where(labels_gpu > 0, image_gpu, cp.inf)
    
    # Create structure for 6-connectivity (3D cross)
    struct = cp.zeros((3, 3, 3), dtype=cp.bool_)
    struct[1, 1, :] = True
    struct[1, :, 1] = True
    struct[:, 1, 1] = True
    
    # Iterative label propagation
    for iteration in range(max_iterations):
        # Find boundary of labeled regions (where labels touch unlabeled)
        labeled_mask = labels_gpu > 0
        
        # Dilate labeled region to find neighbors
        dilated = gpu_ndimage.binary_dilation(labeled_mask, structure=struct)
        
        # Unlabeled pixels that are neighbors of labeled pixels
        propagation_mask = dilated & ~labeled_mask & mask_gpu
        
        if not cp.any(propagation_mask):
            # No more pixels to label
            break
        
        # For each unlabeled pixel, find the best neighbor label
        # "Best" = neighbor with minimum image value (for -distance transform)
        new_labels = labels_gpu.copy()
        new_priority = priority_gpu.copy()
        
        # Track the best (minimum) neighbor image value for each propagation pixel
        best_neighbor_value = cp.full(image_gpu.shape, cp.inf, dtype=cp.float32)
        
        # Check all 6 neighbors
        for axis in range(3):
            for direction in [-1, 1]:
                # Shift labels and image to get neighbor values
                shifted_labels = cp.roll(labels_gpu, direction, axis=axis)
                shifted_image = cp.roll(image_gpu, direction, axis=axis)
                
                # Only consider where we have unlabeled pixels and labeled neighbors
                valid = propagation_mask & (shifted_labels > 0)
                
                if cp.any(valid):
                    # Update if this neighbor has smaller image value than current best
                    better = valid & (shifted_image < best_neighbor_value)
                    new_labels = cp.where(better, shifted_labels, new_labels)
                    new_priority = cp.where(better, shifted_image, new_priority)
                    best_neighbor_value = cp.where(better, shifted_image, best_neighbor_value)
        
        # Count changes
        changes = cp.sum(new_labels != labels_gpu)
        labels_gpu = new_labels
        priority_gpu = new_priority
        
        if iteration % 50 == 0 and iteration > 0:
            print(f"[GPU] watershed iteration {iteration}, changes: {int(changes)}")
        
        if changes == 0:
            break
    
    elapsed = time.time() - start
    print(f"[GPU] watershed: {elapsed:.2f}s, {iteration+1} iterations")
    
    result = backend.to_cpu(labels_gpu)
    
    del image_gpu, labels_gpu, mask_gpu, struct, priority_gpu
    backend.clear_memory()
    
    return result
