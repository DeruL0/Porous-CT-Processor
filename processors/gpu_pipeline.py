"""
"""GPU Pipeline for unified segmentation (EDT -> local_maxima -> watershed)."""

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
    """GPU-accelerated segmentation: EDT -> local_maxima -> watershed."""
    
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
    print(f"[GPU Pipeline] Starting ({size_mb:.1f}MB)")
    
    # Upload
    mask_gpu = cp.asarray(pores_mask.astype(np.uint8))
    
    # EDT
    start = time.time()
    distance_gpu = gpu_ndimage.distance_transform_edt(mask_gpu).astype(cp.float32)
    del mask_gpu
    print(f"[GPU Pipeline] EDT: {time.time() - start:.2f}s")
    
    # Local maxima
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
    
    candidate_values = distance_gpu[is_peak]
    del is_peak
    
    sort_idx = cp.argsort(-candidate_values)
    sorted_candidates_cpu = cp.asnumpy(candidates[sort_idx])
    del candidates, candidate_values, sort_idx
    
    selected_peaks = _nms_cpu(sorted_candidates_cpu, min_peak_distance)
    print(f"[GPU Pipeline] Local maxima: {time.time() - start:.2f}s, {len(selected_peaks)} peaks")
    
    # Markers
    markers_gpu = cp.zeros(pores_mask.shape, dtype=cp.int32)
    if len(selected_peaks) > 0:
        peak_coords = np.array(selected_peaks)
        markers_gpu[tuple(peak_coords.T)] = cp.arange(len(selected_peaks), dtype=cp.int32) + 1
    
    # Watershed
    start = time.time()
    image_gpu = -distance_gpu
    mask_bool_gpu = distance_gpu > 0
    distance_map_cpu = cp.asnumpy(distance_gpu)
    del distance_gpu
    
    segmented_gpu = _watershed_gpu_inplace(image_gpu, markers_gpu, mask_bool_gpu)
    del image_gpu, markers_gpu, mask_bool_gpu
    print(f"[GPU Pipeline] Watershed: {time.time() - start:.2f}s")
    
    # Download
    segmented_cpu = cp.asnumpy(segmented_gpu)
    del segmented_gpu
    
    num_pores = int(np.max(segmented_cpu))
    print(f"[GPU Pipeline] Total: {time.time() - start_total:.2f}s, {num_pores} pores")
    
    # Only free unused blocks instead of all blocks
    backend.clear_memory(force=False)
    
    return distance_map_cpu, segmented_cpu, num_pores


def _nms_cpu(sorted_candidates: np.ndarray, min_distance: int) -> list:
    """Non-maximum suppression with adaptive GPU/KDTree acceleration."""
    n = len(sorted_candidates)
    
    if n > 50000 and CUPY_AVAILABLE:
        try:
            return _nms_gpu_parallel(sorted_candidates, min_distance)
        except Exception:
            pass
    
    if n > 10000:
        from scipy.spatial import cKDTree
        tree = cKDTree(sorted_candidates)
        suppressed = np.zeros(n, dtype=bool)
        selected = []
        for i in range(n):
            if suppressed[i]:
                continue
            peak = sorted_candidates[i]
            selected.append(peak)
            neighbors = tree.query_ball_point(peak, r=min_distance, p=2)
            for j in neighbors:
                if j > i:
                    suppressed[j] = True
        return selected
    suppressed = np.zeros(n, dtype=bool)
    min_dist_sq = min_distance * min_distance
    selected = []
    for i in range(n):
        if suppressed[i]:
            continue
        peak = sorted_candidates[i]
        selected.append(peak)
        if i + 1 < n:
            diff = sorted_candidates[i+1:] - peak
            suppressed[i+1:][np.sum(diff * diff, axis=1) < min_dist_sq] = True
    return selected


def _nms_gpu_parallel(candidates: np.ndarray, min_distance: int) -> list:
    """GPU parallel NMS for large candidate sets."""
    import cupy as cp
    
    backend = get_gpu_backend()
    min_dist_sq = min_distance * min_distance
    candidates_gpu = cp.asarray(candidates, dtype=cp.float32)
    suppressed = cp.zeros(len(candidates), dtype=cp.bool_)
    selected = []
    
    for idx in range(len(candidates)):
        if suppressed[idx].item():
            continue
        selected.append(candidates[idx])
        if idx + 1 < len(candidates):
            diff = candidates_gpu[idx+1:] - candidates_gpu[idx]
            suppressed[idx+1:] |= (cp.sum(diff * diff, axis=1) < min_dist_sq)
    
    del candidates_gpu, suppressed
    backend.clear_memory(force=False)
    return selected


def _watershed_gpu_inplace(image_gpu, markers_gpu, mask_gpu, max_iterations: int = 500):
    """GPU watershed with optimized RawKernel."""
    import cupy as cp
    
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
        
        if (!mask[idx]) { labels_out[idx] = 0; return; }
        int current = labels_in[idx];
        if (current > 0) { labels_out[idx] = current; return; }
        
        int d = shape[0], h = shape[1], w = shape[2];
        int z = idx / (h * w), rem = idx % (h * w), y = rem / w, x = rem % w;
        
        int best_label = 0;
        float best_val = 1e30f;
        
        // 6-connectivity neighbor check (unrolled)
        if (z > 0) {
            int n_idx = (z-1) * h * w + y * w + x;
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
                }
            }
        }
        if (z < d-1) {
            int n_idx = (z+1) * h * w + y * w + x;
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
                }
            }
        }
        if (y > 0) {
            int n_idx = z * h * w + (y-1) * w + x;
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
                }
            }
        }
        if (y < h-1) {
            int n_idx = z * h * w + (y+1) * w + x;
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
                }
            }
        }
        if (x > 0) {
            int n_idx = z * h * w + y * w + (x-1);
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
                }
            }
        }
        if (x < w-1) {
            int n_idx = z * h * w + y * w + (x+1);
            int n_label = labels_in[n_idx];
            if (n_label > 0) {
                float n_val = image[n_idx];
                if (n_val < best_val || (n_val == best_val && n_label < best_label)) {
                    best_val = n_val; best_label = n_label;
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
    
    check_interval, last_changed = 20, -1
    
    for iteration in range(max_iterations):
        watershed_kernel((blocks,), (threads,), (image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed))
        labels_curr, labels_next = labels_next, labels_curr
        
        if iteration % check_interval == check_interval - 1:
            n_changed = changed.item()
            if n_changed == 0:
                break
            if last_changed > 0 and n_changed < last_changed * 0.1:
                check_interval = max(5, check_interval // 2)
            last_changed = n_changed
            changed.fill(0)
    
    del labels_next, shape_gpu, changed
    return labels_curr
