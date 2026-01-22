"""
Unified GPU-accelerated threshold and statistics computation.

Provides a single-pass GPU pipeline for histogram, statistics, and Otsu threshold
to minimize PCIe data transfers.
"""

import numpy as np
import time
from typing import Optional

from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def compute_threshold_stats_gpu(data: np.ndarray, nbins: int = 256) -> dict:
    """
    Unified GPU computation for histogram, statistics, and Otsu threshold.
    
    Performs all computations in a single GPU pass to avoid redundant 
    PCIe transfers. This replaces the separate calls to:
    - compute_histogram_gpu
    - compute_statistics_gpu  
    - threshold_otsu_gpu
    
    Args:
        data: 1D array of pixel values
        nbins: Number of histogram bins
        
    Returns:
        Dictionary with:
        - 'histogram': (hist, bin_edges) tuple
        - 'stats': dict with mean, std, skewness, kurtosis, min, max, n
        - 'otsu_threshold': optimal Otsu threshold value
    """
    if not GPU_ENABLED or not CUPY_AVAILABLE:
        return _compute_threshold_stats_cpu(data, nbins)
    
    backend = get_gpu_backend()
    size_mb = data.nbytes / (1024 * 1024)
    
    if not backend.available or size_mb < GPU_MIN_SIZE_MB:
        return _compute_threshold_stats_cpu(data, nbins)
    
    try:
        import cupy as cp
        
        start = time.time()
        
        # === SINGLE UPLOAD ===
        data_gpu = cp.asarray(data)
        
        # === HISTOGRAM (reused for Otsu) ===
        hist_gpu, bin_edges_gpu = cp.histogram(data_gpu, bins=nbins)
        bin_centers_gpu = (bin_edges_gpu[:-1] + bin_edges_gpu[1:]) / 2
        
        # === STATISTICS ===
        n = int(data_gpu.size)
        mean = float(cp.mean(data_gpu).item())
        std = float(cp.std(data_gpu).item())
        data_min = float(cp.min(data_gpu).item())
        data_max = float(cp.max(data_gpu).item())
        
        if std > 0:
            standardized = (data_gpu - mean) / std
            skewness = float(cp.mean(standardized ** 3).item())
            kurtosis = float(cp.mean(standardized ** 4).item()) - 3
            del standardized
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Data no longer needed after this
        del data_gpu
        
        # === OTSU THRESHOLD (using already-computed histogram) ===
        hist_float = hist_gpu.astype(cp.float64)
        hist_norm = hist_float / hist_float.sum()
        
        # Cumulative sums
        weight1 = cp.cumsum(hist_norm)
        weight2 = 1.0 - weight1
        
        # Cumulative means
        mean1_cumsum = cp.cumsum(hist_norm * bin_centers_gpu)
        mean1 = mean1_cumsum / (weight1 + 1e-10)
        
        total_mean = float((hist_norm * bin_centers_gpu).sum().item())
        mean2 = (total_mean - mean1_cumsum) / (weight2 + 1e-10)
        
        # Inter-class variance
        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # Find threshold that maximizes variance
        idx = int(cp.argmax(variance_between).item())
        otsu_threshold = float(bin_centers_gpu[idx].item())
        
        # === SINGLE DOWNLOAD ===
        hist = cp.asnumpy(hist_gpu)
        bin_edges = cp.asnumpy(bin_edges_gpu)
        
        # Cleanup
        del hist_gpu, bin_edges_gpu, bin_centers_gpu, hist_float, hist_norm
        del weight1, weight2, mean1_cumsum, mean1, mean2, variance_between
        backend.clear_memory()
        
        elapsed = time.time() - start
        print(f"[GPU] compute_threshold_stats: {elapsed:.3f}s (unified pass)")
        
        return {
            'histogram': (hist, bin_edges),
            'stats': {
                'mean': mean,
                'std': std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min': data_min,
                'max': data_max,
                'n': n
            },
            'otsu_threshold': otsu_threshold
        }
        
    except Exception as e:
        print(f"[GPU] compute_threshold_stats failed: {e}, using CPU")
        backend.clear_memory()
        return _compute_threshold_stats_cpu(data, nbins)


def _compute_threshold_stats_cpu(data: np.ndarray, nbins: int = 256) -> dict:
    """CPU fallback for unified threshold/stats computation."""
    from skimage.filters import threshold_otsu
    
    start = time.time()
    
    # Histogram
    hist, bin_edges = np.histogram(data, bins=nbins)
    
    # Statistics
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
    
    # Otsu threshold
    otsu_threshold = threshold_otsu(data)
    
    elapsed = time.time() - start
    print(f"[CPU] compute_threshold_stats: {elapsed:.3f}s")
    
    return {
        'histogram': (hist, bin_edges),
        'stats': {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': data_min,
            'max': data_max,
            'n': n
        },
        'otsu_threshold': otsu_threshold
    }
