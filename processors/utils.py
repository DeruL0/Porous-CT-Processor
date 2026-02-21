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
    
    # Sort by value descending — single argsort, reused for both arrays
    values = image_gpu[is_peak]
    del max_filtered, is_peak
    sort_idx              = cp.argsort(-values)
    candidates_gpu_sorted = candidates[sort_idx]   # (N, 3) int64, descending intensity
    values_gpu_sorted     = values[sort_idx]       # (N,)   float32
    del candidates, values, image_gpu

    # GPU NMS via spatial hash grid  O(N)
    keep_mask  = _nms_gpu_parallel(candidates_gpu_sorted, values_gpu_sorted, min_distance)
    result_gpu = candidates_gpu_sorted[keep_mask]
    result     = backend.to_cpu(result_gpu)
    del candidates_gpu_sorted, values_gpu_sorted, sort_idx, keep_mask, result_gpu
    backend.clear_memory(force=False)
    print(f"[GPU] find_local_maxima: {time.time() - start:.2f}s, {len(result)} peaks")
    return result


# ---------------------------------------------------------------------------
# GPU Non-Maximum Suppression — O(N) via spatial hashing
# ---------------------------------------------------------------------------

def _nms_gpu_parallel(candidates_gpu, intensities_gpu, min_distance: int):
    """
    GPU-accelerated NMS with spatial hash grid.  O(N) vs. O(N²) brute-force.

    Parameters
    ----------
    candidates_gpu  : cp.ndarray (N, 3) int32 / int64  — voxel coordinates
    intensities_gpu : cp.ndarray (N,)  float32          — intensity at each candidate
    min_distance    : int                               — exclusion radius (voxels)

    Returns
    -------
    keep : cp.ndarray (N,) bool   — True where the point IS kept
    """
    import cupy as cp

    n = int(candidates_gpu.shape[0])
    if n == 0:
        return cp.zeros(0, dtype=cp.bool_)

    cell_size   = max(float(min_distance), 1.0)
    pts_f32     = candidates_gpu.astype(cp.float32)           # (N,3)
    grid_coords = cp.floor(pts_f32 / cell_size).astype(cp.int32)  # (N,3)

    # Large coprime primes for spatial hashing
    P1 = np.int64(73_856_093)
    P2 = np.int64(19_349_663)
    P3 = np.int64(83_492_791)
    hash_table_size = max(n * 4, 1024)

    gc_i64      = grid_coords.astype(cp.int64)
    raw_hashes  = (
        gc_i64[:, 0] * int(P1) ^
        gc_i64[:, 1] * int(P2) ^
        gc_i64[:, 2] * int(P3)
    ) % hash_table_size
    raw_hashes  = raw_hashes.astype(cp.int32)

    # Sort by hash bucket to enable contiguous cell ranges
    sort_order      = cp.argsort(raw_hashes)
    sorted_hashes   = raw_hashes[sort_order]
    sorted_pts_f32  = pts_f32[sort_order]          # (N, 3) float32
    sorted_intens   = intensities_gpu[sort_order].astype(cp.float32)
    sorted_gc       = grid_coords[sort_order].astype(cp.int32)  # (N, 3)

    # Build cell_start / cell_end arrays
    cell_start = cp.full(hash_table_size, -1, dtype=cp.int32)
    cell_end   = cp.zeros(hash_table_size, dtype=cp.int32)
    is_boundary = cp.concatenate([
        cp.array([True]),
        sorted_hashes[1:] != sorted_hashes[:-1],
        cp.array([True]),
    ])
    starts        = cp.where(is_boundary[:-1])[0].astype(cp.int32)
    ends          = cp.where(is_boundary[1:])[0].astype(cp.int32)
    bucket_keys   = sorted_hashes[starts]
    cell_start[bucket_keys] = starts
    cell_end[bucket_keys]   = ends + 1

    # ------------------------------------------------------------------
    # RawKernel: each thread owns one candidate; checks 27 neighbor cells
    # ------------------------------------------------------------------
    nms_kernel_code = r'''
    extern "C" __global__
    void nms_spatial_hash_kernel(
        const float* __restrict__ points,      // (N, 3) float32, row-major
        const float* __restrict__ intensities, // (N,)   float32
        const int*   __restrict__ gc,          // (N, 3) int32  grid coords
        const int*   __restrict__ cell_start,  // (hash_table_size,)
        const int*   __restrict__ cell_end,    // (hash_table_size,)
        bool*                     keep_flag,   // (N,)   output
        int          N,
        int          hash_table_size,
        float        min_dist_sq,
        long long    P1,
        long long    P2,
        long long    P3
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        float my_val = intensities[i];
        float p_z    = points[i * 3 + 0];
        float p_y    = points[i * 3 + 1];
        float p_x    = points[i * 3 + 2];
        int   g_z    = gc[i * 3 + 0];
        int   g_y    = gc[i * 3 + 1];
        int   g_x    = gc[i * 3 + 2];

        // Traverse 27 neighbouring grid cells (3x3x3 cube around owning cell)
        for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int cz = g_z + dz;
            int cy = g_y + dy;
            int cx = g_x + dx;
            long long h = ((long long)cz * P1 ^ (long long)cy * P2 ^ (long long)cx * P3)
                          % (long long)hash_table_size;
            if (h < 0) h += (long long)hash_table_size;
            int bucket = (int)h;
            int cs = cell_start[bucket];
            if (cs < 0) continue;
            int ce = cell_end[bucket];
            for (int j = cs; j < ce; j++) {
                if (j == i) continue;
                // Hash-collision guard: verify exact grid-cell match
                if (gc[j*3+0] != cz || gc[j*3+1] != cy || gc[j*3+2] != cx) continue;
                float ddz  = points[j*3+0] - p_z;
                float ddy  = points[j*3+1] - p_y;
                float ddx  = points[j*3+2] - p_x;
                float dist2 = ddz*ddz + ddy*ddy + ddx*ddx;
                if (dist2 < min_dist_sq) {
                    float jval = intensities[j];
                    // Suppress self if neighbour is strictly stronger,
                    // or equally strong with a lower sorted index (tie-break).
                    if (jval > my_val || (jval == my_val && j < i)) {
                        keep_flag[i] = false;
                        return;
                    }
                }
            }
        }}}
        keep_flag[i] = true;
    }
    '''
    nms_kernel = cp.RawKernel(nms_kernel_code, 'nms_spatial_hash_kernel')

    keep_sorted = cp.ones(n, dtype=cp.bool_)
    threads     = 256
    blocks_nms  = (n + threads - 1) // threads

    nms_kernel(
        (blocks_nms,), (threads,),
        (
            sorted_pts_f32.ravel(),
            sorted_intens,
            sorted_gc.ravel(),
            cell_start,
            cell_end,
            keep_sorted,
            np.int32(n),
            np.int32(hash_table_size),
            np.float32(min_distance * min_distance),
            P1, P2, P3,
        ),
    )

    # Map keep flags from sorted-by-hash space back to original candidate order
    keep_original = cp.empty(n, dtype=cp.bool_)
    keep_original[sort_order] = keep_sorted
    return keep_original


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
    """
    GPU watershed — 3-D shared-memory kernel with hierarchical warp/block
    reduction, plus batched host-sync to minimise PCIe stalls.

    Optimisation summary
    --------------------
    * 8x8x8 thread blocks load a (10,10,10) halo tile into __shared__ memory
      so all 6-neighbour reads hit L1 instead of L2/DRAM.
    * atomicAdd on changed_flag is replaced by:
        thread-local int  →  warp __shfl_down_sync  →  block __shared__ int
        →  single atomicAdd per block (27-64× fewer global atomic ops).
    * Host loop synchronises (calls .item()) only every CHECK_INTERVAL steps,
      keeping the GPU pipeline full between convergence checks.
    """
    import cupy as cp

    backend = get_gpu_backend()
    start   = time.time()

    watershed_kernel_code = r'''
    #define BLOCK_DIM_X 8
    #define BLOCK_DIM_Y 8
    #define BLOCK_DIM_Z 8

    extern "C" __global__
    void watershed_kernel(
        const float* __restrict__ image,      // [d*h*w] float32
        const int*   __restrict__ labels_in,  // [d*h*w] int32
        int*                      labels_out, // [d*h*w] int32
        const bool*  __restrict__ mask,       // [d*h*w] bool  (may be NULL)
        const int*   __restrict__ shape,      // [3]  = {d, h, w}
        int*                      changed_flag
    ) {
        // ---- 3-D shared memory with 1-pixel halo ---------------------------
        __shared__ int   s_labels[BLOCK_DIM_Z+2][BLOCK_DIM_Y+2][BLOCK_DIM_X+2];
        __shared__ float s_image [BLOCK_DIM_Z+2][BLOCK_DIM_Y+2][BLOCK_DIM_X+2];
        __shared__ int   s_block_changed;

        const int d = shape[0], h = shape[1], w = shape[2];

        // Global voxel this thread "owns"
        int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        int gy = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        int gz = blockIdx.z * BLOCK_DIM_Z + threadIdx.z;

        // Flat thread ID within the block (for cooperative loading & warp ID)
        int tid = threadIdx.z * (BLOCK_DIM_Y * BLOCK_DIM_X)
                + threadIdx.y * BLOCK_DIM_X
                + threadIdx.x;

        if (tid == 0) s_block_changed = 0;
        __syncthreads();

        // ---- Cooperative halo load -----------------------------------------
        // Shared tile size: (BZ+2)*(BY+2)*(BX+2) = 1000 elements
        // Block threads:    8*8*8 = 512  → each thread loads 1-2 elements
        const int BX2 = BLOCK_DIM_X + 2;
        const int BY2 = BLOCK_DIM_Y + 2;
        const int BZ2 = BLOCK_DIM_Z + 2;
        const int shared_total = BX2 * BY2 * BZ2;
        const int block_threads = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

        // Origin of tile in global space (includes -1 halo offset)
        int bx0 = (int)blockIdx.x * BLOCK_DIM_X - 1;
        int by0 = (int)blockIdx.y * BLOCK_DIM_Y - 1;
        int bz0 = (int)blockIdx.z * BLOCK_DIM_Z - 1;

        for (int si = tid; si < shared_total; si += block_threads) {
            int sx = si % BX2;
            int sy = (si / BX2) % BY2;
            int sz = si / (BX2 * BY2);
            int rx = bx0 + sx;
            int ry = by0 + sy;
            int rz = bz0 + sz;
            if (rx >= 0 && rx < w && ry >= 0 && ry < h && rz >= 0 && rz < d) {
                int g = rz * h * w + ry * w + rx;
                s_labels[sz][sy][sx] = labels_in[g];
                s_image [sz][sy][sx] = image[g];
            } else {
                s_labels[sz][sy][sx] = 0;
                s_image [sz][sy][sx] = 1e30f;
            }
        }
        __syncthreads();

        // ---- Process owned voxel  ------------------------------------------
        int local_changed = 0;

        if (gx < w && gy < h && gz < d) {
            int g_idx   = gz * h * w + gy * w + gx;
            bool masked = mask ? mask[g_idx] : true;

            if (!masked) {
                labels_out[g_idx] = 0;
            } else {
                // Shared-memory indices (+1 for halo offset)
                int sx = threadIdx.x + 1;
                int sy = threadIdx.y + 1;
                int sz = threadIdx.z + 1;

                int cur = s_labels[sz][sy][sx];
                if (cur > 0) {
                    labels_out[g_idx] = cur;
                } else {
                    int   best_label = 0;
                    float best_val   = 1e30f;

                    #define CHECK_SHM(dz, dy, dx) {\
                        int nl = s_labels[sz+(dz)][sy+(dy)][sx+(dx)]; \
                        if (nl > 0) { \
                            float nv = s_image[sz+(dz)][sy+(dy)][sx+(dx)]; \
                            if (nv < best_val || (nv == best_val && nl < best_label)) { \
                                best_val = nv; best_label = nl; \
                            } \
                        } \
                    }
                    CHECK_SHM(-1,  0,  0)
                    CHECK_SHM(+1,  0,  0)
                    CHECK_SHM( 0, -1,  0)
                    CHECK_SHM( 0, +1,  0)
                    CHECK_SHM( 0,  0, -1)
                    CHECK_SHM( 0,  0, +1)
                    #undef CHECK_SHM

                    if (best_label > 0) {
                        labels_out[g_idx] = best_label;
                        local_changed = 1;
                    } else {
                        labels_out[g_idx] = 0;
                    }
                }
            }
        }

        // ---- Hierarchical reduction: thread → warp → block → global --------
        // Warp-level reduction with shuffle
        unsigned int full_mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_changed += __shfl_down_sync(full_mask, local_changed, offset);

        // Warp lane 0 accumulates into shared block counter
        if ((tid & 31) == 0)
            atomicAdd(&s_block_changed, local_changed);
        __syncthreads();

        // Block thread 0 does single global atomic per block
        if (tid == 0 && s_block_changed > 0)
            atomicAdd(changed_flag, s_block_changed);
    }
    '''
    watershed_kernel = cp.RawKernel(watershed_kernel_code, 'watershed_kernel')

    d, h, w = image.shape
    image_gpu   = cp.asarray(image, dtype=cp.float32)
    labels_curr = cp.asarray(markers, dtype=cp.int32)
    labels_next = cp.empty_like(labels_curr)
    mask_gpu    = (cp.asarray(mask, dtype=cp.bool_)
                   if mask is not None
                   else cp.ones(image.shape, dtype=cp.bool_))
    shape_gpu   = cp.array([d, h, w], dtype=cp.int32)
    changed     = cp.zeros(1, dtype=cp.int32)

    BLOCK = (8, 8, 8)
    GRID  = ((w + 7) // 8, (h + 7) // 8, (d + 7) // 8)

    # -------------------------------------------------------------------
    # Batched host-sync: run CHECK_INTERVAL kernel launches between each
    # .item() call so the GPU pipeline is never starved waiting for the
    # host to decide "keep going".
    # -------------------------------------------------------------------
    CHECK_INTERVAL = 10
    final_iter     = 0
    n_changed      = 1  # sentinel — assume not converged yet

    for outer in range(0, max_iterations, CHECK_INTERVAL):
        changed.fill(0)  # async clear — no host stall
        for inner in range(CHECK_INTERVAL):
            current_iter = outer + inner
            if current_iter >= max_iterations:
                break
            watershed_kernel(
                GRID, BLOCK,
                (image_gpu, labels_curr, labels_next, mask_gpu, shape_gpu, changed),
            )
            labels_curr, labels_next = labels_next, labels_curr
            final_iter = current_iter

        # Single sync per batch — amortises PCIe round-trip over CHECK_INTERVAL iters
        n_changed = int(changed.item())
        if n_changed == 0:
            break

    print(f"[GPU] watershed: {time.time() - start:.2f}s, {final_iter + 1} iters")
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
