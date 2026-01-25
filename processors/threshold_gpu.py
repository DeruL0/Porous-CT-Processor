"""Unified GPU threshold and statistics computation."""

import numpy as np
import time
from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def compute_threshold_stats_gpu(data: np.ndarray, nbins: int = 256) -> dict:
    """Unified GPU: histogram, statistics, and Otsu threshold in single pass."""
    if not GPU_ENABLED or not CUPY_AVAILABLE:
        return _compute_threshold_stats_cpu(data, nbins)
    
    backend = get_gpu_backend()
    if not backend.available or data.nbytes / (1024 * 1024) < GPU_MIN_SIZE_MB:
        return _compute_threshold_stats_cpu(data, nbins)
    
    try:
        import cupy as cp
        start = time.time()
        
        data_gpu = cp.asarray(data)
        hist_gpu, bin_edges_gpu = cp.histogram(data_gpu, bins=nbins)
        bin_centers = (bin_edges_gpu[:-1] + bin_edges_gpu[1:]) / 2
        
        # Statistics
        n = int(data_gpu.size)
        mean, std = float(cp.mean(data_gpu).item()), float(cp.std(data_gpu).item())
        data_min, data_max = float(cp.min(data_gpu).item()), float(cp.max(data_gpu).item())
        
        skewness, kurtosis = 0.0, 0.0
        if std > 0:
            z = (data_gpu - mean) / std
            skewness, kurtosis = float(cp.mean(z**3).item()), float(cp.mean(z**4).item()) - 3
        del data_gpu
        
        # Otsu threshold
        hist_norm = hist_gpu.astype(cp.float64) / hist_gpu.sum()
        w1, w2 = cp.cumsum(hist_norm), 1.0 - cp.cumsum(hist_norm)
        m1_cs = cp.cumsum(hist_norm * bin_centers)
        m1 = m1_cs / (w1 + 1e-10)
        m2 = (float((hist_norm * bin_centers).sum().item()) - m1_cs) / (w2 + 1e-10)
        otsu = float(bin_centers[cp.argmax(w1 * w2 * (m1 - m2)**2)].item())
        
        hist, bin_edges = cp.asnumpy(hist_gpu), cp.asnumpy(bin_edges_gpu)
        del hist_gpu, bin_edges_gpu, bin_centers, hist_norm, w1, w2, m1_cs, m1, m2
        backend.clear_memory(force=False)
        
        print(f"[GPU] threshold_stats: {time.time() - start:.3f}s")
        return {
            'histogram': (hist, bin_edges),
            'stats': {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis,
                      'min': data_min, 'max': data_max, 'n': n},
            'otsu_threshold': otsu
        }
    except Exception as e:
        print(f"[GPU] threshold_stats failed: {e}")
        backend.clear_memory(force=False)
        return _compute_threshold_stats_cpu(data, nbins)


def _compute_threshold_stats_cpu(data: np.ndarray, nbins: int = 256) -> dict:
    """CPU fallback for threshold/stats."""
    from skimage.filters import threshold_otsu
    start = time.time()
    
    hist, bin_edges = np.histogram(data, bins=nbins)
    mean, std = float(np.mean(data)), float(np.std(data))
    data_min, data_max = float(np.min(data)), float(np.max(data))
    
    skewness, kurtosis = 0.0, 0.0
    if std > 0:
        z = (data - mean) / std
        skewness, kurtosis = float(np.mean(z**3)), float(np.mean(z**4)) - 3
    
    print(f"[CPU] threshold_stats: {time.time() - start:.3f}s")
    return {
        'histogram': (hist, bin_edges),
        'stats': {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis,
                  'min': data_min, 'max': data_max, 'n': len(data)},
        'otsu_threshold': threshold_otsu(data)
    }
