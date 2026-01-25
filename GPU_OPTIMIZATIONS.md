# GPU优化文档

优化日期: 2026年1月26日

## 优化清单

| 优化项 | 文件 | 性能提升 |
|--------|------|---------|
| Watershed Kernel展开 | gpu_pipeline.py, utils.py | 15-25% |
| 内存清理策略 | gpu_backend.py | 10-20% |
| GPU并行NMS | gpu_pipeline.py | 50-100% (>50k点) |
| 自适应收敛检查 | gpu_pipeline.py, utils.py | 5-15% |
| CuPy内存池配置 | gpu_backend.py | 提高稳定性 |
| DICOM Rescale优化 | dicom_utils.py | 10-20% |
| 统一Threshold计算 | utils.py, threshold_gpu.py | 50% |

## 最佳实践

```python
# 推荐: 统一threshold计算
from processors.threshold_gpu import compute_threshold_stats_gpu
result = compute_threshold_stats_gpu(data)

# 推荐: 使用统一pipeline
from processors.gpu_pipeline import run_segmentation_pipeline_gpu
dist, seg, n = run_segmentation_pipeline_gpu(mask)

# 内存清理
backend.clear_memory(force=False)  # 默认: 只释放未使用块
backend.clear_memory(force=True)   # 仅在内存不足时使用
```

## 注意事项

- 所有优化保持向后兼容
- GPU不可用时自动降级到CPU
- 小数据集(<10MB)仍使用CPU
