"""GPU-accelerated utility functions with CPU fallback."""

import numpy as np
import scipy.ndimage as ndimage
from typing import Optional
import time

from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def binary_fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask (GPU-accelerated)."""
    if not GPU_ENABLED:
        return ndimage.binary_fill_holes(binary_mask)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and backend.get_free_memory_mb() > size_mb * 4:
        try:
            import cupyx.scipy.ndimage as gpu_ndimage
            start = time.time()
            mask_gpu = backend.to_gpu(binary_mask)
            result = backend.to_cpu(gpu_ndimage.binary_fill_holes(mask_gpu))
            del mask_gpu
            backend.clear_memory(force=False)
            print(f"[GPU] binary_fill_holes: {time.time() - start:.2f}s")
            return result
        except Exception as e:
            print(f"[GPU] binary_fill_holes failed: {e}")
            backend.clear_memory(force=False)
    
    start = time.time()
    result = ndimage.binary_fill_holes(binary_mask)
    print(f"[CPU] binary_fill_holes: {time.time() - start:.2f}s")
    return result


def distance_transform_edt(binary_mask: np.ndarray, 
                           sampling: Optional[tuple] = None) -> np.ndarray:
    """Euclidean Distance Transform (GPU-accelerated)."""
    if not GPU_ENABLED:
        return ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    
    backend = get_gpu_backend()
    size_mb = binary_mask.nbytes / (1024 * 1024)
    required_mb = size_mb * 32
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb:
        try:
            import cupyx.scipy.ndimage as gpu_ndimage
            start = time.time()
            mask_gpu = backend.to_gpu(binary_mask.astype(np.uint8))
            result = backend.to_cpu(gpu_ndimage.distance_transform_edt(mask_gpu, sampling=sampling)).astype(np.float32)
            del mask_gpu
            backend.clear_memory(force=False)
            print(f"[GPU] EDT: {time.time() - start:.2f}s")
            return result
        except Exception as e:
            print(f"[GPU] EDT failed: {e}")
            backend.clear_memory(force=False)
    elif backend.available and size_mb >= GPU_MIN_SIZE_MB:
        print(f"[GPU] EDT skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB")

    start = time.time()
    result = ndimage.distance_transform_edt(binary_mask, sampling=sampling).astype(np.float32)
    print(f"[CPU] EDT: {time.time() - start:.2f}s")
    return result


def find_local_maxima(image: np.ndarray,
                      min_distance: int = 1,
                      labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Find local maxima with GPU acceleration and NMS."""
    from skimage.feature import peak_local_max
    
    if not GPU_ENABLED:
        return peak_local_max(image, min_distance=min_distance, labels=labels)
    
    backend = get_gpu_backend()
    size_mb = image.nbytes / (1024 * 1024)
    required_mb = size_mb * 4
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8 and CUPY_AVAILABLE:
        try:
            return _find_local_maxima_gpu_impl(image, min_distance, labels)
        except Exception as e:
            print(f"[GPU] find_local_maxima failed: {e}")
            backend.clear_memory()
    elif backend.available and size_mb >= GPU_MIN_SIZE_MB:
        print(f"[GPU] find_local_maxima skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB")
    
    start = time.time()
    result = peak_local_max(image, min_distance=min_distance, labels=labels)
    print(f"[CPU] find_local_maxima: {time.time() - start:.2f}s, {len(result)} peaks")
    return result


def _find_local_maxima_gpu_impl(image: np.ndarray,
                                 min_distance: int,
                                 labels: Optional[np.ndarray]) -> np.ndarray:
    """GPU local maxima detection with NMS."""
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    
    backend = get_gpu_backend()
    start = time.time()
    
    image_gpu = cp.asarray(image.astype(np.float32))
    size = 2 * min_distance + 1
    max_filtered = gpu_ndimage.maximum_filter(image_gpu, size=size)
    is_peak = (image_gpu == max_filtered) & (image_gpu > 0)
    
    if labels is not None:
        is_peak &= cp.asarray(labels) > 0
    
    candidates = cp.argwhere(is_peak)
    if candidates.shape[0] == 0:
        del image_gpu, max_filtered, is_peak
        backend.clear_memory(force=False)
        return np.array([]).reshape(0, image.ndim)
    
    # Sort by value descending
    values = image_gpu[is_peak]
    del max_filtered, is_peak
    sorted_candidates = backend.to_cpu(candidates[cp.argsort(-values)])
    del candidates, values, image_gpu
    backend.clear_memory(force=False)
    
    # NMS
    n = len(sorted_candidates)
    if n > 10000:
        # Use KDTree for large candidate sets
        from scipy.spatial import cKDTree
        tree = cKDTree(sorted_candidates)
        suppressed = np.zeros(n, dtype=bool)
        selected_peaks = []
        for i in range(n):
            if suppressed[i]:
                continue
            selected_peaks.append(sorted_candidates[i])
            for j in tree.query_ball_point(sorted_candidates[i], r=min_distance, p=2):
                if j > i:
                    suppressed[j] = True
    else:
        selected_peaks = []
        suppressed = np.zeros(n, dtype=bool)
        min_dist_sq = min_distance * min_distance
        for i in range(n):
            if suppressed[i]:
                continue
            selected_peaks.append(sorted_candidates[i])
            if i + 1 < n:
                diff = sorted_candidates[i+1:] - sorted_candidates[i]
                suppressed[i+1:][np.sum(diff * diff, axis=1) < min_dist_sq] = True
    
    result = np.array(selected_peaks) if selected_peaks else np.array([]).reshape(0, image.ndim)
    print(f"[GPU] find_local_maxima: {time.time() - start:.2f}s, {len(result)} peaks")
    return result


def watershed_gpu(image: np.ndarray,
                  markers: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  max_iterations: int = 500) -> np.ndarray:
    """GPU-accelerated watershed segmentation."""
    from skimage.segmentation import watershed as cpu_watershed
    
    if not GPU_ENABLED:
        return cpu_watershed(image, markers, mask=mask)
    
    backend = get_gpu_backend()
    size_mb = image.nbytes / (1024 * 1024)
    required_mb = size_mb * 5
    free_mb = backend.get_free_memory_mb()
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and required_mb < free_mb * 0.8 and CUPY_AVAILABLE:
        try:
            return _watershed_gpu_impl(image, markers, mask, max_iterations)
        except Exception as e:
            print(f"[GPU] watershed failed: {e}")
            backend.clear_memory()
    elif backend.available and size_mb >= GPU_MIN_SIZE_MB:
        print(f"[GPU] watershed skipped: need {required_mb:.0f}MB, have {free_mb:.0f}MB")
    
    start = time.time()
    result = cpu_watershed(image, markers, mask=mask)
    print(f"[CPU] watershed: {time.time() - start:.2f}s")
    return result


def _watershed_gpu_impl(image: np.ndarray,
                        markers: np.ndarray,
                        mask: Optional[np.ndarray],
                        max_iterations: int) -> np.ndarray:
    """GPU watershed with optimized RawKernel."""
    import cupy as cp
    
    backend = get_gpu_backend()
    start = time.time()
    
    watershed_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void watershed_kernel(
        const float* image,         // 0
        const int* labels_in,       // 1
        int* labels_out,            // 2
        const bool* mask,           // 3
        const int* shape,           // 4
        int* changed_flag           // 5
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total_pixels = shape[0] * shape[1] * shape[2];
        
        if (idx >= total_pixels) return;
        
        bool is_masked = mask ? mask[idx] : true;
        int current_label = labels_in[idx];
        
        if (!is_masked) { labels_out[idx] = 0; return; }
        if (current_label > 0) { labels_out[idx] = current_label; return; }
        
        int d = shape[0], h = shape[1], w = shape[2];
        int z = idx / (h * w);
        int rem = idx % (h * w);
        int y = rem / w;
        int x = rem % w;
        
        int best_label = 0;
        float best_neighbor_val = 1e30;
        
        #define CHECK_NEIGHBOR(nz, ny, nx) \
            if ((nz) >= 0 && (nz) < d && (ny) >= 0 && (ny) < h && (nx) >= 0 && (nx) < w) { \
                int n_idx = (nz) * h * w + (ny) * w + (nx); \
                int n_label = labels_in[n_idx]; \
                if (n_label > 0) { \
                    float n_val = image[n_idx]; \
                    if (n_val < best_neighbor_val || (n_val == best_neighbor_val && n_label < best_label)) { \
                        best_neighbor_val = n_val; best_label = n_label; \
                    } \
                } \
            }
        
        CHECK_NEIGHBOR(z-1, y, x)
        CHECK_NEIGHBOR(z+1, y, x)
        CHECK_NEIGHBOR(z, y-1, x)
        CHECK_NEIGHBOR(z, y+1, x)
        CHECK_NEIGHBOR(z, y, x-1)
        CHECK_NEIGHBOR(z, y, x+1)
        
        #undef CHECK_NEIGHBOR
        
        if (best_label > 0) {
            labels_out[idx] = best_label;
            atomicAdd(changed_flag, 1);
        } else {
            labels_out[idx] = 0;
        }
    }
    ''', 'watershed_kernel')
    
    image_gpu = cp.asarray(image, dtype=cp.float32)
    labels_curr = cp.asarray(markers, dtype=cp.int32)
    labels_next = cp.empty_like(labels_curr)
    mask_gpu = cp.asarray(mask, dtype=cp.bool_) if mask is not None else cp.ones(image.shape, dtype=cp.bool_)
    shape_gpu = cp.array(image.shape, dtype=cp.int32)
    
    total_pixels = image.size
    blocks = (total_pixels + 255) // 256
    changed = cp.zeros(1, dtype=cp.int32)
    check_interval, last_changed = 20, -1
    
    for iteration in range(max_iterations):
        watershed_kernel((blocks,), (256,), (image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed))
        labels_curr, labels_next = labels_next, labels_curr
        
        if iteration % check_interval == check_interval - 1:
            n_changed = changed.item()
            if n_changed == 0:
                break
            if last_changed > 0 and n_changed < last_changed * 0.1:
                check_interval = max(5, check_interval // 2)
            last_changed = n_changed
            changed.fill(0)
    
    print(f"[GPU] watershed: {time.time() - start:.2f}s, {iteration+1} iters")
    result = backend.to_cpu(labels_curr)
    del image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed
    backend.clear_memory(force=False)
    return result


# =============================================================================
# GPU-Accelerated Threshold Computation Functions
# =============================================================================

def compute_histogram_gpu(data: np.ndarray, bins: int = 256, 
                          range_: Optional[tuple] = None) -> tuple:
    """GPU-accelerated histogram computation."""
    if not GPU_ENABLED:
        return np.histogram(data, bins=bins, range=range_)
    
    backend = get_gpu_backend()
    size_mb = data.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE:
        try:
            import cupy as cp
            start = time.time()
            data_gpu = cp.asarray(data)
            hist_gpu, bin_edges_gpu = cp.histogram(data_gpu, bins=bins, range=range_)
            hist, bin_edges = cp.asnumpy(hist_gpu), cp.asnumpy(bin_edges_gpu)
            del data_gpu, hist_gpu, bin_edges_gpu
            backend.clear_memory(force=False)
            print(f"[GPU] histogram: {time.time() - start:.3f}s")
            return hist, bin_edges
        except Exception as e:
            print(f"[GPU] histogram failed: {e}")
            backend.clear_memory(force=False)
    
    start = time.time()
    result = np.histogram(data, bins=bins, range=range_)
    print(f"[CPU] histogram: {time.time() - start:.3f}s")
    return result


def compute_statistics_gpu(data: np.ndarray) -> dict:
    """GPU-accelerated statistics: mean, std, skewness, kurtosis."""
    if not GPU_ENABLED:
        return _compute_statistics_cpu(data)
    
    backend = get_gpu_backend()
    if not (backend.available and data.nbytes / (1024 * 1024) >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE):
        return _compute_statistics_cpu(data)
    
    try:
        import cupy as cp
        start = time.time()
        data_gpu = cp.asarray(data)
        
        mean, std = float(cp.mean(data_gpu).item()), float(cp.std(data_gpu).item())
        data_min, data_max = float(cp.min(data_gpu).item()), float(cp.max(data_gpu).item())
        
        skewness, kurtosis = 0.0, 0.0
        if std > 0:
            z = (data_gpu - mean) / std
            skewness, kurtosis = float(cp.mean(z**3).item()), float(cp.mean(z**4).item()) - 3
        
        del data_gpu
        backend.clear_memory(force=False)
        print(f"[GPU] statistics: {time.time() - start:.3f}s")
        
        return {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis,
                'min': data_min, 'max': data_max, 'n': len(data)}
    except Exception as e:
        print(f"[GPU] statistics failed: {e}")
        backend.clear_memory(force=False)
        return _compute_statistics_cpu(data)


def _compute_statistics_cpu(data: np.ndarray) -> dict:
    """CPU statistics computation."""
    start = time.time()
    mean, std = float(np.mean(data)), float(np.std(data))
    data_min, data_max = float(np.min(data)), float(np.max(data))
    
    skewness, kurtosis = 0.0, 0.0
    if std > 0:
        z = (data - mean) / std
        skewness, kurtosis = float(np.mean(z**3)), float(np.mean(z**4)) - 3
    
    print(f"[CPU] statistics: {time.time() - start:.3f}s")
    return {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis,
            'min': data_min, 'max': data_max, 'n': len(data)}


def threshold_otsu_gpu(data: np.ndarray, nbins: int = 256) -> float:
    """GPU-accelerated Otsu threshold (delegates to unified function)."""
    try:
        from processors.threshold_gpu import compute_threshold_stats_gpu
        return compute_threshold_stats_gpu(data, nbins)['otsu_threshold']
    except Exception:
        pass
    
    from skimage.filters import threshold_otsu
    if not GPU_ENABLED:
        return threshold_otsu(data)
    
    backend = get_gpu_backend()
    if not (backend.available and data.nbytes / (1024 * 1024) >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE):
        return threshold_otsu(data)
    
    try:
        import cupy as cp
        start = time.time()
        data_gpu = cp.asarray(data)
        hist, bin_edges = cp.histogram(data_gpu, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist_norm = hist.astype(cp.float64) / hist.sum()
        w1, w2 = cp.cumsum(hist_norm), 1.0 - cp.cumsum(hist_norm)
        m1_cs = cp.cumsum(hist_norm * bin_centers)
        m1 = m1_cs / (w1 + 1e-10)
        m2 = (float((hist_norm * bin_centers).sum().item()) - m1_cs) / (w2 + 1e-10)
        
        threshold = float(bin_centers[cp.argmax(w1 * w2 * (m1 - m2)**2)].item())
        del data_gpu, hist, bin_edges, bin_centers, hist_norm, w1, w2, m1_cs, m1, m2
        backend.clear_memory(force=False)
        print(f"[GPU] Otsu: {time.time() - start:.3f}s, value={threshold:.1f}")
        return threshold
    except Exception as e:
        print(f"[GPU] Otsu failed: {e}")
        backend.clear_memory(force=False)
        return threshold_otsu(data)
