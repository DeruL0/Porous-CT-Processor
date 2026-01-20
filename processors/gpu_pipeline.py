"""
GPU Pipeline for Pore Network Modeling.

Provides a unified GPU execution path that keeps data on GPU memory
across multiple processing stages:
  binary_fill_holes -> EDT -> local_maxima -> watershed

This avoids the 6+ PCIe round-trips that occur when each function
independently transfers data CPU <-> GPU.
"""

import numpy as np
import time
from typing import Optional, Tuple
import gc

from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def run_segmentation_pipeline_gpu(
    pores_mask: np.ndarray,
    min_peak_distance: int = 6
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    GPU-accelerated segmentation pipeline with minimal data transfers.
    
    Workflow (all on GPU):
    1. Distance transform
    2. Find local maxima
    3. Create markers
    4. Watershed segmentation
    
    Args:
        pores_mask: Binary mask of pore space (NumPy, bool)
        min_peak_distance: Minimum distance between pore centers
        
    Returns:
        Tuple of (distance_map, segmented_regions, num_pores)
        All returned as NumPy arrays (transferred from GPU at end)
    """
    if not GPU_ENABLED or not CUPY_AVAILABLE:
        # Fall back to individual CPU functions
        from processors.utils import distance_transform_edt, find_local_maxima
        from skimage.segmentation import watershed
        
        distance_map = distance_transform_edt(pores_mask)
        local_maxi = find_local_maxima(distance_map, min_distance=min_peak_distance, labels=pores_mask)
        
        markers = np.zeros(pores_mask.shape, dtype=np.int32)
        if len(local_maxi) > 0:
            markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
        
        segmented = watershed(-distance_map, markers, mask=distance_map > 0)
        num_pores = int(np.max(segmented))
        
        return distance_map, segmented, num_pores
    
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    
    backend = get_gpu_backend()
    size_mb = pores_mask.nbytes / (1024 * 1024)
    
    # Check if we have enough GPU memory for the full pipeline
    # Need: mask + distance + markers + labels + workspace ~= 6x input size
    required_mb = size_mb * 6
    free_mb = backend.get_free_memory_mb()
    
    if required_mb > free_mb * 0.8:
        print(f"[GPU Pipeline] Insufficient memory ({required_mb:.0f}MB needed, {free_mb:.0f}MB free), using CPU")
        from processors.utils import distance_transform_edt, find_local_maxima, watershed_gpu
        distance_map = distance_transform_edt(pores_mask)
        local_maxi = find_local_maxima(distance_map, min_distance=min_peak_distance, labels=pores_mask)
        markers = np.zeros(pores_mask.shape, dtype=np.int32)
        if len(local_maxi) > 0:
            markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
        segmented = watershed_gpu(-distance_map, markers, mask=distance_map > 0)
        return distance_map, segmented, int(np.max(segmented))
    
    start_total = time.time()
    print(f"[GPU Pipeline] Starting unified GPU segmentation ({size_mb:.1f}MB)")
    
    # === SINGLE UPLOAD ===
    start = time.time()
    mask_gpu = cp.asarray(pores_mask.astype(np.uint8))
    print(f"[GPU Pipeline] Upload: {time.time() - start:.2f}s")
    
    # === DISTANCE TRANSFORM (on GPU) ===
    start = time.time()
    distance_gpu = gpu_ndimage.distance_transform_edt(mask_gpu).astype(cp.float32)
    del mask_gpu  # Free immediately
    print(f"[GPU Pipeline] EDT: {time.time() - start:.2f}s")
    
    # === FIND LOCAL MAXIMA (on GPU) ===
    start = time.time()
    size = 2 * min_peak_distance + 1
    max_filtered = gpu_ndimage.maximum_filter(distance_gpu, size=size)
    is_peak = (distance_gpu == max_filtered) & (distance_gpu > 0)
    del max_filtered
    
    candidates = cp.argwhere(is_peak)
    n_candidates = candidates.shape[0]
    
    if n_candidates == 0:
        print("[GPU Pipeline] No peaks found")
        backend.clear_memory()
        return np.zeros(pores_mask.shape, dtype=np.float32), np.zeros(pores_mask.shape, dtype=np.int32), 0
    
    # Get values at candidates for sorting
    candidate_values = distance_gpu[is_peak]
    del is_peak
    
    # Sort by value descending
    sort_idx = cp.argsort(-candidate_values)
    sorted_candidates = candidates[sort_idx]
    del candidates, candidate_values, sort_idx
    
    # Transfer sorted candidates to CPU for NMS (sequential algorithm)
    sorted_candidates_cpu = cp.asnumpy(sorted_candidates)
    del sorted_candidates
    
    # NMS on CPU (inherently sequential)
    selected_peaks = _nms_cpu(sorted_candidates_cpu, min_peak_distance)
    print(f"[GPU Pipeline] Local maxima: {time.time() - start:.2f}s, {len(selected_peaks)} peaks")
    
    # === CREATE MARKERS (on GPU) ===
    markers_gpu = cp.zeros(pores_mask.shape, dtype=cp.int32)
    if len(selected_peaks) > 0:
        peak_coords = np.array(selected_peaks)
        markers_gpu[tuple(peak_coords.T)] = cp.arange(len(selected_peaks), dtype=cp.int32) + 1
    
    # === WATERSHED (on GPU using RawKernel) ===
    start = time.time()
    image_gpu = -distance_gpu  # Negative for watershed
    mask_bool_gpu = distance_gpu > 0
    
    # Keep distance_map for return, but as CPU array
    distance_map_cpu = cp.asnumpy(distance_gpu)
    del distance_gpu
    
    segmented_gpu = _watershed_gpu_inplace(image_gpu, markers_gpu, mask_bool_gpu)
    del image_gpu, markers_gpu, mask_bool_gpu
    print(f"[GPU Pipeline] Watershed: {time.time() - start:.2f}s")
    
    # === SINGLE DOWNLOAD ===
    start = time.time()
    segmented_cpu = cp.asnumpy(segmented_gpu)
    del segmented_gpu
    backend.clear_memory()
    print(f"[GPU Pipeline] Download: {time.time() - start:.2f}s")
    
    num_pores = int(np.max(segmented_cpu))
    print(f"[GPU Pipeline] Total: {time.time() - start_total:.2f}s, {num_pores} pores")
    
    return distance_map_cpu, segmented_cpu, num_pores


def _nms_cpu(sorted_candidates: np.ndarray, min_distance: int) -> list:
    """Non-maximum suppression (must be sequential, runs on CPU)."""
    if len(sorted_candidates) > 10000:
        from scipy.spatial import cKDTree
        tree = cKDTree(sorted_candidates)
        suppressed = np.zeros(len(sorted_candidates), dtype=bool)
        selected = []
        for i in range(len(sorted_candidates)):
            if suppressed[i]:
                continue
            peak = sorted_candidates[i]
            selected.append(peak)
            neighbors = tree.query_ball_point(peak, r=min_distance, p=2)
            for j in neighbors:
                if j > i:
                    suppressed[j] = True
        return selected
    else:
        suppressed = np.zeros(len(sorted_candidates), dtype=bool)
        min_dist_sq = min_distance * min_distance
        selected = []
        for i in range(len(sorted_candidates)):
            if suppressed[i]:
                continue
            peak = sorted_candidates[i]
            selected.append(peak)
            if i + 1 < len(sorted_candidates):
                remaining = sorted_candidates[i+1:]
                diff = remaining - peak
                dist_sq = np.sum(diff * diff, axis=1)
                suppressed[i+1:][dist_sq < min_dist_sq] = True
        return selected


def _watershed_gpu_inplace(image_gpu, markers_gpu, mask_gpu, max_iterations: int = 500):
    """
    GPU watershed that operates on existing GPU arrays (no transfer).
    Uses the optimized RawKernel implementation.
    """
    import cupy as cp
    
    # Compile kernel
    watershed_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void watershed_kernel(
        const float* image,
        const int* labels_in,
        int* labels_out,
        const bool* mask,
        const int* shape,
        int* changed_flag
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total = shape[0] * shape[1] * shape[2];
        if (idx >= total) return;
        
        bool is_masked = mask[idx];
        int current = labels_in[idx];
        
        if (!is_masked) { labels_out[idx] = 0; return; }
        if (current > 0) { labels_out[idx] = current; return; }
        
        int d = shape[0], h = shape[1], w = shape[2];
        int z = idx / (h * w);
        int rem = idx % (h * w);
        int y = rem / w;
        int x = rem % w;
        
        int best_label = 0;
        float best_val = 1e30f;
        
        int dz[] = {-1, 1, 0, 0, 0, 0};
        int dy[] = {0, 0, -1, 1, 0, 0};
        int dx[] = {0, 0, 0, 0, -1, 1};
        
        for (int i = 0; i < 6; i++) {
            int nz = z + dz[i], ny = y + dy[i], nx = x + dx[i];
            if (nz >= 0 && nz < d && ny >= 0 && ny < h && nx >= 0 && nx < w) {
                int n_idx = nz * h * w + ny * w + nx;
                int n_label = labels_in[n_idx];
                if (n_label > 0) {
                    float n_val = image[n_idx];
                    if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                        best_val = n_val;
                        best_label = n_label;
                    }
                }
            }
        }
        
        if (best_label > 0) {
            labels_out[idx] = best_label;
            atomicAdd(changed_flag, 1);
        } else {
            labels_out[idx] = 0;
        }
    }
    ''', 'watershed_kernel')
    
    shape_gpu = cp.array(image_gpu.shape, dtype=cp.int32)
    labels_curr = markers_gpu.astype(cp.int32)
    labels_next = cp.empty_like(labels_curr)
    changed = cp.zeros(1, dtype=cp.int32)
    
    total = image_gpu.size
    threads = 256
    blocks = (total + threads - 1) // threads
    
    for iteration in range(max_iterations):
        watershed_kernel(
            (blocks,), (threads,),
            (image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed)
        )
        labels_curr, labels_next = labels_next, labels_curr
        
        if iteration % 10 == 9:
            if changed.item() == 0:
                break
            changed.fill(0)
    
    del labels_next, shape_gpu, changed
    return labels_curr
