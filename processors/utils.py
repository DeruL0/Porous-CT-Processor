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
    
    # Check empty using shape (still syncs but more explicit)
    n_candidates = candidates.shape[0]
    if n_candidates == 0:
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
    Optimized GPU watershed using a single custom RawKernel for propagation.
    
    This replaces the Python loop (which launched ~15 kernels per iteration)
    with a single kernel launch per iteration + one reduction for convergence check.
    
    Algorithm: Priority-Flood (Iterative)
    """
    import cupy as cp
    
    backend = get_gpu_backend()
    start = time.time()
    
    # Pre-compile kernel
    # Use extern "C" to ensure function name is preserved
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
        
        // If not in mask or already labeled, just copy and return
        bool is_masked = mask ? mask[idx] : true;
        int current_label = labels_in[idx];
        
        if (!is_masked) {
            labels_out[idx] = 0;
            return;
        }
        
        if (current_label > 0) {
            labels_out[idx] = current_label;
            return;
        }
        
        // Coordinate calculation
        int d = shape[0];
        int h = shape[1];
        int w = shape[2];
        
        int z = idx / (h * w);
        int rem = idx % (h * w);
        int y = rem / w;
        int x = rem % w;
        
        // Check 6-neighbors
        float my_val = image[idx];
        int best_label = 0;
        float best_neighbor_val = 1e30; // Infinity
        
        // Offsets for 6-connectivity: z-1, z+1, y-1, y+1, x-1, x+1
        int dz[] = {-1, 1, 0, 0, 0, 0};
        int dy[] = {0, 0, -1, 1, 0, 0};
        int dx[] = {0, 0, 0, 0, -1, 1};
        
        for (int i = 0; i < 6; i++) {
            int nz = z + dz[i];
            int ny = y + dy[i];
            int nx = x + dx[i];
            
            if (nz >= 0 && nz < d && ny >= 0 && ny < h && nx >= 0 && nx < w) {
                int n_idx = nz * h * w + ny * w + nx;
                int n_label = labels_in[n_idx];
                
                if (n_label > 0) {
                    float n_val = image[n_idx];
                    
                    // We want the neighbor with the SMALLEST image value 
                    // (since input is -distance_map, smallest = effectively highest distance)
                    // Or if values are equal, pick consistently (e.g. smallest label) to avoid flickering
                    if (n_val < best_neighbor_val) {
                        best_neighbor_val = n_val;
                        best_label = n_label;
                    } else if (n_val == best_neighbor_val) {
                         if (best_label == 0 || n_label < best_label) {
                             best_label = n_label;
                         }
                    }
                }
            }
        }
        
        // If we found a valid neighbor
        if (best_label > 0) {
            labels_out[idx] = best_label;
            // Atomic add to signal change (just need > 0)
            atomicAdd(changed_flag, 1);
        } else {
            labels_out[idx] = 0;
        }
    }
    ''', 'watershed_kernel')
    
    # Data Transfer
    # Ensure float32 for consistency
    image_gpu = cp.asarray(image, dtype=cp.float32)
    labels_curr = cp.asarray(markers, dtype=cp.int32)
    labels_next = cp.empty_like(labels_curr)
    
    if mask is not None:
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)
    else:
        mask_gpu = None  # Kernel handles null check via raw pointer logic (trickier in Python) but we'll pass None or array
        # Easier to pass array of ones if None, to keep kernel signature simple
        mask_gpu = cp.ones(image.shape, dtype=cp.bool_)

    shape_gpu = cp.array(image.shape, dtype=cp.int32)
    
    # Kernel config
    total_pixels = image.size
    threads_per_block = 256
    blocks = (total_pixels + threads_per_block - 1) // threads_per_block
    
    # Iteration loop
    changed = cp.zeros(1, dtype=cp.int32)
    
    iteration = 0
    for iteration in range(max_iterations):
        # Launch kernel
        watershed_kernel(
            (blocks,), (threads_per_block,),
            (image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed)
        )
        
        # Swap buffers
        labels_curr, labels_next = labels_next, labels_curr
        
        # Check convergence every 10 iterations to reduce sync overhead (P2 fix)
        if iteration % 10 == 9:
            n_changed = changed.item()
            if n_changed == 0:
                break
            changed.fill(0)
            if iteration % 100 == 99:
                print(f"[GPU] watershed iter {iteration+1}, changed: {n_changed}")
    
    elapsed = time.time() - start
    print(f"[GPU] unique kernel watershed: {elapsed:.2f}s, {iteration+1} iterations")
    
    result = backend.to_cpu(labels_curr)
    
    del image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed
    backend.clear_memory()
    
    return result


# =============================================================================
# GPU-Accelerated Threshold Computation Functions
# =============================================================================

def compute_histogram_gpu(data: np.ndarray, bins: int = 256, 
                          range_: Optional[tuple] = None) -> tuple:
    """
    GPU-accelerated histogram computation.
    
    Args:
        data: 1D or flattened array of values
        bins: Number of histogram bins
        range_: Optional (min, max) range for histogram
        
    Returns:
        (hist, bin_edges) tuple, same as np.histogram
    """
    if not GPU_ENABLED:
        return np.histogram(data, bins=bins, range=range_)
    
    backend = get_gpu_backend()
    size_mb = data.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE:
        try:
            import cupy as cp
            
            start = time.time()
            
            # Transfer to GPU
            data_gpu = cp.asarray(data)
            
            # Compute histogram on GPU
            if range_ is not None:
                hist_gpu, bin_edges_gpu = cp.histogram(data_gpu, bins=bins, range=range_)
            else:
                hist_gpu, bin_edges_gpu = cp.histogram(data_gpu, bins=bins)
            
            # Transfer back to CPU
            hist = cp.asnumpy(hist_gpu)
            bin_edges = cp.asnumpy(bin_edges_gpu)
            
            del data_gpu, hist_gpu, bin_edges_gpu
            backend.clear_memory()
            
            print(f"[GPU] histogram: {time.time() - start:.3f}s, {bins} bins")
            return hist, bin_edges
            
        except Exception as e:
            print(f"[GPU] histogram failed: {e}, using CPU")
            backend.clear_memory()
    
    # CPU fallback
    start = time.time()
    result = np.histogram(data, bins=bins, range=range_)
    print(f"[CPU] histogram: {time.time() - start:.3f}s")
    return result


def compute_statistics_gpu(data: np.ndarray) -> dict:
    """
    GPU-accelerated statistical computation: mean, std, skewness, kurtosis.
    
    Args:
        data: 1D or flattened array of values
        
    Returns:
        Dictionary with 'mean', 'std', 'skewness', 'kurtosis', 'min', 'max'
    """
    if not GPU_ENABLED:
        return _compute_statistics_cpu(data)
    
    backend = get_gpu_backend()
    size_mb = data.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE:
        try:
            import cupy as cp
            
            start = time.time()
            
            data_gpu = cp.asarray(data)
            
            n = data_gpu.size
            mean = float(cp.mean(data_gpu).item())
            std = float(cp.std(data_gpu).item())
            data_min = float(cp.min(data_gpu).item())
            data_max = float(cp.max(data_gpu).item())
            
            if std > 0:
                # Standardized data
                standardized = (data_gpu - mean) / std
                skewness = float(cp.mean(standardized ** 3).item())
                kurtosis = float(cp.mean(standardized ** 4).item()) - 3
                del standardized
            else:
                skewness = 0.0
                kurtosis = 0.0
            
            del data_gpu
            backend.clear_memory()
            
            print(f"[GPU] statistics: {time.time() - start:.3f}s")
            
            return {
                'mean': mean,
                'std': std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min': data_min,
                'max': data_max,
                'n': n
            }
            
        except Exception as e:
            print(f"[GPU] statistics failed: {e}, using CPU")
            backend.clear_memory()
    
    return _compute_statistics_cpu(data)


def _compute_statistics_cpu(data: np.ndarray) -> dict:
    """CPU implementation of statistics computation."""
    start = time.time()
    
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data))
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    
    if std > 0:
        standardized = (data - mean) / std
        skewness = float(np.mean(standardized ** 3))
        kurtosis = float(np.mean(standardized ** 4)) - 3
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    print(f"[CPU] statistics: {time.time() - start:.3f}s")
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': data_min,
        'max': data_max,
        'n': n
    }


def threshold_otsu_gpu(data: np.ndarray, nbins: int = 256) -> float:
    """
    GPU-accelerated Otsu threshold computation.
    
    Implements Otsu's method to find optimal threshold that minimizes
    intra-class variance (or equivalently maximizes inter-class variance).
    
    Args:
        data: 1D array of pixel values
        nbins: Number of histogram bins
        
    Returns:
        Optimal threshold value
    """
    if not GPU_ENABLED:
        from skimage.filters import threshold_otsu
        return threshold_otsu(data)
    
    backend = get_gpu_backend()
    size_mb = data.nbytes / (1024 * 1024)
    
    if backend.available and size_mb >= GPU_MIN_SIZE_MB and CUPY_AVAILABLE:
        try:
            import cupy as cp
            
            start = time.time()
            
            data_gpu = cp.asarray(data)
            
            # Compute histogram
            hist, bin_edges = cp.histogram(data_gpu, bins=nbins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Normalize histogram to get probabilities
            hist = hist.astype(cp.float64)
            hist_norm = hist / hist.sum()
            
            # Cumulative sums
            weight1 = cp.cumsum(hist_norm)
            weight2 = 1.0 - weight1
            
            # Cumulative means
            mean1_cumsum = cp.cumsum(hist_norm * bin_centers)
            mean1 = mean1_cumsum / (weight1 + 1e-10)
            
            total_mean = float((hist_norm * bin_centers).sum().item())
            mean2 = (total_mean - mean1_cumsum) / (weight2 + 1e-10)
            
            # Inter-class variance
            variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
            
            # Find threshold that maximizes variance
            idx = int(cp.argmax(variance_between).item())
            threshold = float(bin_centers[idx].item())
            
            del data_gpu, hist, bin_edges, bin_centers, hist_norm
            del weight1, weight2, mean1_cumsum, mean1, mean2, variance_between
            backend.clear_memory()
            
            print(f"[GPU] Otsu threshold: {time.time() - start:.3f}s, value={threshold:.1f}")
            return threshold
            
        except Exception as e:
            print(f"[GPU] Otsu failed: {e}, using CPU")
            backend.clear_memory()
    
    # CPU fallback
    from skimage.filters import threshold_otsu
    start = time.time()
    result = threshold_otsu(data)
    print(f"[CPU] Otsu threshold: {time.time() - start:.3f}s")
    return result
