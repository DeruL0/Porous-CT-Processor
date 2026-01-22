# GPU Implementation Review

> Based on: **cpu_2_gpu Skill** (从 CPU 到 CUDA：基于 CuPy 的代码重构与性能调优指南)

---

## Summary

| Category | Status | Notes |
|----------|--------|-------|
| **数据精度 (FP32)** | ✅ Good | All GPU code uses `float32` / `int32`. |
| **显存预算** | ⚠️ Partial | Memory checks exist, but no batching for huge volumes. |
| **数据闭环** | ❌ Problem | Frequent CPU↔GPU transfers break data locality. |
| **隐式同步** | ⚠️ Partial | Several `.item()` calls in loops cause sync stalls. |
| **核函数预热** | ✅ Good | `gpu_backend.py` implements warmup. |
| **向量化** | ✅ Good | Watershed kernel is fully vectorized. |
| **显存池清理** | ✅ Good | `clear_memory()` called after operations. |

---

## Detailed Findings

### 1. 数据闭环 (Data Locality) - P1: HIGH IMPACT

**Problem**: The current workflow is:

```
CPU NumPy → GPU (fill_holes) → CPU → GPU (EDT) → CPU → GPU (watershed) → CPU
```

Each function independently transfers data CPU → GPU → CPU. This causes **6+ PCIe round-trips** per PNM processing.

**Files Affected**:

- [processors/utils.py](file:///d:/Projects/6.Food%20CT/Porous/processors/utils.py) lines 15-107 (binary_fill_holes, distance_transform_edt)
- [processors/pnm.py](file:///d:/Projects/6.Food%20CT/Porous/processors/pnm.py) lines 166-276 (`_run_segmentation`)

**Recommendation**:
Refactor to a "GPU Pipeline" design:

1. Transfer data to GPU **once** at the start of the pipeline.
2. Keep intermediate results (`pores_mask_gpu`, `distance_map_gpu`) on GPU.
3. Only transfer final `segmented_regions` back to CPU at the end.

```python
# Proposed: processors/gpu_pipeline.py
def run_segmentation_gpu(data: np.ndarray, threshold: int):
    import cupy as cp
    backend = get_gpu_backend()
    
    # Single upload
    data_gpu = backend.to_gpu(data)
    
    solid_mask_gpu = data_gpu > threshold
    filled_gpu = gpu_ndimage.binary_fill_holes(solid_mask_gpu)
    pores_mask_gpu = filled_gpu ^ solid_mask_gpu
    del solid_mask_gpu, filled_gpu
    
    distance_gpu = gpu_ndimage.distance_transform_edt(pores_mask_gpu)
    # ... watershed_gpu on GPU ...
    
    # Single download
    return backend.to_cpu(segmented_gpu)
```

---

### 2. 隐式同步 (Implicit Sync) - P2: MEDIUM IMPACT

**Problem**: In `_watershed_gpu_impl`, every iteration calls `changed.item()` which forces a GPU→CPU sync and stalls the pipeline.

**File**: [processors/utils.py#L470](file:///d:/Projects/6.Food%20CT/Porous/processors/utils.py#L470)

```python
if changed.item() == 0:  # Implicit sync every iteration!
    break
```

**Recommendation**:
Check convergence less frequently (e.g., every 10-20 iterations) to reduce sync overhead:

```python
if iteration % 10 == 0:
    if changed.item() == 0:
        break
    changed.fill(0)
```

---

### 3. 分批处理 (Batching) - P2: MEDIUM IMPACT

**Problem**: If a volume exceeds VRAM, the current code falls back to CPU entirely. No chunked/tiled GPU processing exists.

**Files Affected**:

- [processors/utils.py](file:///d:/Projects/6.Food%20CT/Porous/processors/utils.py) lines 79, 139, 307

**Recommendation**:
For EDT and watershed on large volumes, implement Z-slab batching:

```python
for z in range(0, depth, slab_size):
    slab_gpu = backend.to_gpu(volume[z:z+slab_size])
    result_slab = process_slab_gpu(slab_gpu)
    result[z:z+slab_size] = backend.to_cpu(result_slab)
    del slab_gpu, result_slab
    backend.clear_memory()
```

---

### 4. Pinned Memory 未使用 - P3: LOW IMPACT

**Problem**: `gpu_backend.py` has `to_gpu_async` but never uses pinned memory for faster PCIe transfers.

**Recommendation**:
For large transfers during DICOM loading, allocate pinned host memory:

```python
# In DicomSeriesLoader._build_volume
pinned_buffer = cp.cuda.alloc_pinned_memory(volume.nbytes)
pinned_array = np.frombuffer(pinned_buffer, dtype=np.float32).reshape(volume.shape)
# ... read DICOM slices into pinned_array ...
volume_gpu = cp.asarray(pinned_array)  # Fast DMA transfer
```

---

### 5. rescale_volume_gpu 未集成 - P2: MEDIUM IMPACT

**Problem**: The new `rescale_volume_gpu` in `dicom_utils.py` is defined but **never called** from `DicomSeriesLoader`.

**File**: [loaders/dicom_utils.py#L47](file:///d:/Projects/6.Food%20CT/Porous/loaders/dicom_utils.py#L47)

**Recommendation**:
Integrate into the loader pipeline when GPU is enabled:

```python
# In DicomSeriesLoader._build_volume
if GPU_ENABLED:
    volume_gpu = cp.asarray(volume)
    rescale_volume_gpu(volume_gpu, slopes, intercepts)
    volume = cp.asnumpy(volume_gpu)
else:
    rescale_volume_numba(volume, slopes, intercepts)
```

---

## Priority Summary

| Priority | Issue | Effort |
|----------|-------|--------|
| P1 | Data Locality (GPU Pipeline) | High |
| P2 | Implicit Sync in Watershed | Low |
| P2 | GPU Batching for Large Volumes | Medium |
| P2 | Integrate `rescale_volume_gpu` | Low |
| P3 | Pinned Memory | Low |

---

## Recommendations for Next Steps

1. **Short-term (P2 fixes)**: Integrate `rescale_volume_gpu` and reduce sync frequency in watershed.
2. **Medium-term (P1)**: Create a unified `processors/gpu_pipeline.py` that keeps data on GPU across fill_holes → EDT → watershed.
3. **Long-term (P3)**: Implement pinned memory for DICOM loading to maximize PCIe bandwidth.

---

## Threshold Computation Review

### Files Analyzed

- [processors/utils.py](file:///d:/Projects/6.Food%20CT/Porous/processors/utils.py): `compute_histogram_gpu`, `compute_statistics_gpu`, `threshold_otsu_gpu`
- [processors/pore.py](file:///d:/Projects/6.Food%20CT/Porous/processors/pore.py): `suggest_threshold`

### Current Workflow

```
suggest_threshold()
├── compute_histogram_gpu(clean_data)     # Upload data → histogram → Download
├── compute_statistics_gpu(clean_data)    # Upload data → stats → Download  
└── threshold_otsu_gpu(clean_data)        # Upload data → Otsu → Download
```

### Issues Identified

| Issue | Severity | Description |
| ----- | -------- | ----------- |
| **Multiple Uploads** | P1 | Same `clean_data` uploaded to GPU 3 times |
| **Separate Downloads** | P2 | Results downloaded separately instead of in batch |
| **No Data Reuse** | P1 | `threshold_otsu_gpu` recomputes histogram internally |

### Detailed Analysis

#### P1: Data Locality Violation

The `suggest_threshold` function in `pore.py` calls three GPU functions sequentially:

1. `compute_histogram_gpu(clean_data)` - uploads `clean_data`
2. `compute_statistics_gpu(clean_data)` - uploads `clean_data` again
3. `threshold_otsu_gpu(clean_data)` - uploads `clean_data` a third time

**Impact**: 3x unnecessary PCIe transfers for the same data.

#### P1: Redundant Histogram Computation

`threshold_otsu_gpu` computes its own histogram internally (line 668):

```python
hist, bin_edges = cp.histogram(data_gpu, bins=nbins)
```

But `compute_histogram_gpu` already computed a histogram, which is used only for UI purposes in `suggest_threshold`.

### Recommendations

#### Option A: Unified Threshold Pipeline (Recommended)

Create `compute_threshold_stats_gpu()` that does everything in one GPU pass:

```python
def compute_threshold_stats_gpu(data: np.ndarray) -> dict:
    """Single GPU pass for histogram, statistics, and Otsu threshold."""
    import cupy as cp
    
    # Single upload
    data_gpu = cp.asarray(data)
    
    # Compute histogram (reused for Otsu)
    hist, bin_edges = cp.histogram(data_gpu, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute statistics
    mean = float(cp.mean(data_gpu).item())
    std = float(cp.std(data_gpu).item())
    # ... skewness, kurtosis ...
    
    # Compute Otsu using already-computed histogram
    # ... Otsu calculation ...
    
    # Single download
    return {
        'histogram': (cp.asnumpy(hist), cp.asnumpy(bin_edges)),
        'stats': {'mean': mean, 'std': std, ...},
        'otsu_threshold': threshold
    }
```

#### Option B: Keep GPU Data Alive

Modify functions to optionally accept/return GPU arrays:

```python
def compute_histogram_gpu(data, bins=256, keep_on_gpu=False):
    data_gpu = cp.asarray(data) if isinstance(data, np.ndarray) else data
    hist_gpu, edges_gpu = cp.histogram(data_gpu, bins=bins)
    if keep_on_gpu:
        return hist_gpu, edges_gpu, data_gpu  # Return GPU data for reuse
    return cp.asnumpy(hist_gpu), cp.asnumpy(edges_gpu)
```

### Priority Summary

| Priority | Fix | Effort |
| -------- | --- | ------ |
| P1 | Create unified `compute_threshold_stats_gpu` | Medium |
| P1 | Reuse histogram in Otsu calculation | Low |
| P2 | Batch result downloads | Low |
