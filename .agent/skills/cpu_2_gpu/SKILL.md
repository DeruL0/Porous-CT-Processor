# 从 CPU 到 CUDA：基于 CuPy 的代码重构与性能调优指南

将 CPU 程序（通常基于 NumPy）迁移至 GPU（基于 CuPy）不仅仅是更换库名，更是一场关于“数据移动控制”和“计算密度提升”的重构。

## 1. 架构思维的转变：Host vs. Device

在 CPU 编程中，内存是透明的；在 CUDA 编程中，必须时刻意识到 **Host（CPU+内存）** 与 **Device（GPU+显存）** 之间有一条名为 PCIe 的“窄桥”。

### 核心原则

- **计算密度**：只有当计算量远大于数据传输量时，使用 GPU 才有意义。

- **数据闭环**：尽量让数据留在 GPU 内部完成所有计算流程，减少中途返回 CPU 的次数。

## 2. 显存容量预算与管理

显存（VRAM）通常比系统内存小得多且昂贵，溢出（OOM）是迁移后的第一个拦路虎。

### A. 精度降级：从 `float64` 到 `float32`

- **注意事项**：CPU 默认常用 `double` (64bit)，但消费级显卡的单精度（FP32）算力远高于双精度（FP64）。

- **建议**：除非对数值精度有极高要求（如精密物理模拟），否则一律使用 `dtype=cp.float32`。这能立刻减少 50% 的显存占用并提升数倍速度。

### B. 显存占用的数学估算

在编写程序前，养成估算显存占用量的习惯：

- **公式**：`显存 (GB) = 元素总数 * 每个元素的字节数 / 1024^3`

- **案例**：一个 $20000 \times 20000$ 的 `float32` 矩阵占用：$4 \times 10^8 \times 4 \text{ bytes} \approx 1.49 \text{ GB}$。

## 3. 数据分批 (Batching) 与传输优化

当数据集总量（如 50GB）超过单卡显存（如 24GB）时，必须采用分批策略。

### A. 分片计算 (Tiling / Batching)

不要试图一次性将整个大数据集推向 GPU。通过循环将数据切片，并在每个批次结束时及时释放引用。

```
import cupy as cp

# 假设 data_cpu 是一个巨大的 NumPy 数组
batch_size = 10000 
results_cpu = []

for i in range(0, len(data_cpu), batch_size):
    # 1. 仅上传当前批次
    batch_gpu = cp.array(data_cpu[i : i + batch_size])
    
    # 2. GPU 密集计算
    res_gpu = complex_kernel(batch_gpu)
    
    # 3. 结果传回 CPU 并立即释放 GPU 变量
    results_cpu.append(res_gpu.get())
    
    # 手动删除引用有助于显存池更快回收
    del batch_gpu, res_gpu
```

### B. 异步传输：使用 CUDA Stream

默认情况下，数据传输和计算是顺序阻塞的。使用 `cp.cuda.Stream` 可以实现并行：

- **流流水线**：在 GPU 计算 Batch N 的同时，通过异步流将 Batch N+1 从内存传向显存。这可以有效掩盖 PCIe 的传输耗时。

## 4. 关键避坑：避免“性能杀手”

### A. 隐式数据同步

某些看起来无害的 Python 操作会导致严重的性能回退，因为它们强迫 GPU 停止工作并把数据传回 CPU。

- **避坑指南**：

  - 不要在循环内 `print(gpu_array)` 或使用 `if gpu_array.max() > 0:`。

  - 不要在循环内使用 `matplotlib` 绘制显存中的数组。

  - **正确做法**：使用 CuPy 原生的逻辑运算函数（如 `cp.where`, `cp.any`）在显存内完成判断。

### B. 核函数预热 (Warm-up)

CUDA 驱动程序在第一次调用特定的核函数时会进行编译和初始化。

- **技巧**：在正式性能测试或大规模生产计算前，先用一组小规模的伪数据运行一遍程序。

## 5. 性能调优清单 (Checklist)

**检查项**

**操作目标**

**收益**

**向量化**

消除所有显式 `for` 循环，改用矩阵运算

**极高**：利用张量核心并行能力

**合并访存**

确保相邻线程访问相邻显存地址（SoA 结构）

**高**：最大化显存带宽利用率

**显存池**

使用 `mempool.free_all_blocks()` 手动清理

**中**：防止长时间运行后的碎片化 OOM

**Pins Memory**

在 CPU 端使用固定内存 (Pinned Memory)

**中**：提升 `cp.array()` 和 `.get()` 的传输速度

## 总结

将 CPU 代码修改为 CUDA 程序的精髓在于：**少搬运、多计算、大批量、低精度。** 通过 CuPy 提供的显存池和流管理工具，你可以在保持 Python 开发效率的同时，获得接近原声 C++ CUDA 的性能体验。
