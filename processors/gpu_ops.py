"""
GPU-accelerated operations for image processing.
Provides transparent fallback to CPU when GPU is unavailable.
"""

import numpy as np
from typing import Optional
import time

from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage


def binary_fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask with GPU acceleration.
    
    Args:
        binary_mask: Binary 3D array
        
    Returns:
        Filled binary mask
    """
    if not GPU_ENABLED:
        import scipy.ndimage as ndimage
        return ndimage.binary_fill_holes(binary_mask)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    
    # Use GPU only for larger arrays
    if backend.available and size_mb >= GPU_MIN_SIZE_MB:
        try:
            mask_gpu = backend.to_gpu(binary_mask.astype(np.bool_))
            result_gpu = gpu_ndimage.binary_fill_holes(mask_gpu)
            result = backend.to_cpu(result_gpu)
            del mask_gpu, result_gpu
            return result
        except Exception:
            pass  # Fall through to CPU
    
    # CPU fallback
    import scipy.ndimage as ndimage
    return ndimage.binary_fill_holes(binary_mask)


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
        import scipy.ndimage as ndimage
        return ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    
    # EDT needs ~8x memory (float64 intermediate arrays)
    required_mb = size_mb * 8
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8:
        try:
            start = time.time()
            mask_gpu = backend.to_gpu(binary_mask.astype(np.uint8))
            
            # CuPy's distance_transform_edt
            if sampling is not None:
                result_gpu = gpu_ndimage.distance_transform_edt(mask_gpu, sampling=sampling)
            else:
                result_gpu = gpu_ndimage.distance_transform_edt(mask_gpu)
            
            result = backend.to_cpu(result_gpu).astype(np.float32)
            del mask_gpu, result_gpu
            backend.clear_memory()
            
            elapsed = time.time() - start
            print(f"[GPU] distance_transform_edt: {elapsed:.2f}s")
            return result
        except Exception as e:
            print(f"[GPU] EDT failed ({e}), using CPU")
            backend.clear_memory()
    else:
        if backend.available:
            print(f"[GPU] EDT skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB free")
    
    # CPU fallback
    import scipy.ndimage as ndimage
    start = time.time()
    result = ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    elapsed = time.time() - start
    print(f"[CPU] distance_transform_edt: {elapsed:.2f}s")
    return result


def find_local_maxima(image: np.ndarray,
                      min_distance: int = 1,
                      labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Find local maxima with GPU acceleration.
    
    Args:
        image: Input image
        min_distance: Minimum distance between peaks
        labels: Optional label image to restrict search
        
    Returns:
        Array of peak coordinates (N, ndim)
    """
    if not GPU_ENABLED:
        from skimage.feature import peak_local_max
        return peak_local_max(image, min_distance=min_distance, labels=labels)
    
    backend = get_gpu_backend()
    size_mb = image.nbytes / (1024 * 1024)
    
    # Need ~3x memory: image + max_filtered + is_peak + labels
    required_mb = size_mb * 4 if labels is not None else size_mb * 3
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8:
        try:
            start = time.time()
            image_gpu = backend.to_gpu(image.astype(np.float32))
            
            # Use maximum_filter to find local maxima
            size = 2 * min_distance + 1
            max_filtered = gpu_ndimage.maximum_filter(image_gpu, size=size)
            
            # Peaks: where image equals local max and is > 0
            is_peak = (image_gpu == max_filtered) & (image_gpu > 0)
            
            # Apply labels mask if provided
            if labels is not None:
                labels_gpu = backend.to_gpu(labels)
                is_peak = is_peak & (labels_gpu > 0)
                del labels_gpu
            
            # Get coordinates
            peaks_gpu = cp.argwhere(is_peak)
            peaks = backend.to_cpu(peaks_gpu)
            
            del image_gpu, max_filtered, is_peak, peaks_gpu
            backend.clear_memory()
            
            elapsed = time.time() - start
            print(f"[GPU] find_local_maxima: {elapsed:.2f}s, found {len(peaks)} peaks")
            return peaks
        except Exception as e:
            print(f"[GPU] find_local_maxima failed ({e}), using CPU")
            backend.clear_memory()
    else:
        if backend.available:
            print(f"[GPU] find_local_maxima skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB free")
    
    # CPU fallback
    from skimage.feature import peak_local_max
    start = time.time()
    result = peak_local_max(image, min_distance=min_distance, labels=labels)
    elapsed = time.time() - start
    print(f"[CPU] find_local_maxima: {elapsed:.2f}s, found {len(result)} peaks")
    return result


def compute_all_throat_radii(segmented_regions: np.ndarray, 
                              distance_map: np.ndarray,
                              connections: set,
                              spacing: tuple) -> dict:
    """
    Batch compute all throat radii using vectorized operations.
    
    Args:
        segmented_regions: Watershed segmentation labels (int32)
        distance_map: Distance transform of pore space (float32)
        connections: Set of (pore_a, pore_b) tuples
        spacing: Voxel spacing (sx, sy, sz)
        
    Returns:
        Dict mapping (pore_a, pore_b) -> throat_radius
    """
    if not connections:
        return {}
    
    avg_spacing = sum(spacing) / 3.0
    backend = get_gpu_backend()
    size_mb = segmented_regions.nbytes / (1024 * 1024)
    
    print(f"[Throat] Computing radii for {len(connections)} connections...")
    start = time.time()
    
    # Try GPU if available
    use_gpu = GPU_ENABLED and backend.available and size_mb >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE
    
    if use_gpu:
        try:
            result = _compute_throat_radii_vectorized(
                segmented_regions, distance_map, connections, avg_spacing, use_gpu=True
            )
            elapsed = time.time() - start
            print(f"[Throat] GPU completed: {elapsed:.2f}s, {len(result)}/{len(connections)} measured")
            return result
        except Exception as e:
            print(f"[Throat] GPU failed ({e}), falling back to CPU")
    
    # CPU fallback
    result = _compute_throat_radii_vectorized(
        segmented_regions, distance_map, connections, avg_spacing, use_gpu=False
    )
    elapsed = time.time() - start
    print(f"[Throat] CPU completed: {elapsed:.2f}s, {len(result)}/{len(connections)} measured")
    return result


def _compute_throat_radii_vectorized(labels: np.ndarray,
                                      distances: np.ndarray,
                                      connections: set,
                                      avg_spacing: float,
                                      use_gpu: bool = False) -> dict:
    """
    Vectorized throat measurement - no Python for loops over voxels.
    """
    # Convert connections to a set of sorted tuples for fast lookup
    conn_set = {(min(a, b), max(a, b)) for a, b in connections}
    
    # Collect all boundary data
    all_pairs = []
    all_dists = []
    
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np
    
    if use_gpu:
        backend = get_gpu_backend()
        labels_arr = backend.to_gpu(labels.astype(np.int32))
        dist_arr = backend.to_gpu(distances.astype(np.float32))
    else:
        labels_arr = labels
        dist_arr = distances
    
    print("[Throat] Finding boundaries (6 directions)...", end=" ", flush=True)
    
    for axis in range(3):
        for direction in [1, -1]:
            # Shift labels
            shifted = xp.roll(labels_arr, direction, axis=axis)
            
            # Find boundary voxels
            boundary_mask = (labels_arr > 0) & (shifted > 0) & (labels_arr != shifted)
            
            if not xp.any(boundary_mask):
                continue
            
            # Extract data at boundary
            label_a = labels_arr[boundary_mask]
            label_b = shifted[boundary_mask]
            dist_vals = dist_arr[boundary_mask]
            
            # Ensure a < b for consistent keys
            label_min = xp.minimum(label_a, label_b)
            label_max = xp.maximum(label_a, label_b)
            
            # Transfer to CPU (must use copy to avoid view issues)
            if use_gpu:
                label_min_cpu = backend.to_cpu(label_min)
                label_max_cpu = backend.to_cpu(label_max)
                dist_cpu = backend.to_cpu(dist_vals)
            else:
                label_min_cpu = np.asarray(label_min).copy()
                label_max_cpu = np.asarray(label_max).copy()
                dist_cpu = np.asarray(dist_vals).copy()
            
            # Create pair keys (vectorized)
            # Pack two int32 into int64 for faster grouping
            pairs = label_min_cpu.astype(np.int64) * 1000000 + label_max_cpu.astype(np.int64)
            
            all_pairs.append(pairs.copy())
            all_dists.append(dist_cpu.copy())
            
            del label_a, label_b, dist_vals, label_min, label_max, boundary_mask, shifted
    
    print("done", flush=True)
    
    if use_gpu:
        del labels_arr, dist_arr
        backend.clear_memory()
    
    if not all_pairs:
        return {}
    
    # Concatenate all data
    print("[Throat] Aggregating boundary data...", end=" ", flush=True)
    all_pairs = np.concatenate(all_pairs)
    all_dists = np.concatenate(all_dists)
    print(f"{len(all_pairs)} boundary voxels", flush=True)
    
    # Group by pair and find minimum distance (vectorized using numpy unique)
    print("[Throat] Computing min distances...", end=" ", flush=True)
    
    # Sort by pair for groupby
    sort_idx = np.argsort(all_pairs)
    sorted_pairs = all_pairs[sort_idx]
    sorted_dists = all_dists[sort_idx]
    
    # Find unique pairs and their start indices
    unique_pairs, first_idx = np.unique(sorted_pairs, return_index=True)
    
    # Compute maximum for each group using np.maximum.reduceat
    # Scientific Correctness: Throat radius = Radius of the largest inscribed sphere 
    # that can pass through the interface. This corresponds to the MAXIMUM 
    # distance transform value on the boundary surface.
    max_dists = np.maximum.reduceat(sorted_dists, first_idx)
    
    print("done", flush=True)
    
    # Build result dict, filtering to requested connections only
    print("[Throat] Building result map...", end=" ", flush=True)
    result = {}
    for i, pair_key in enumerate(unique_pairs):
        a = int(pair_key // 1000000)
        b = int(pair_key % 1000000)
        key = (a, b)
        if key in conn_set:
            result[key] = float(max_dists[i]) * avg_spacing
    
    # Debug: check variation in throat radii
    if result:
        radii = list(result.values())
        print(f"{len(result)} mapped, range: {min(radii):.3f} - {max(radii):.3f}", flush=True)
    else:
        print("0 mapped", flush=True)
    
    return result
