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
