"""
GPU Backend Manager for CuPy acceleration.
Provides transparent switching between NumPy and CuPy based on availability.
"""

from typing import Optional, Any, Tuple
import numpy as np

# Try to import CuPy
try:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    gpu_ndimage = None
    CUPY_AVAILABLE = False


class GPUBackend:
    """Singleton for GPU backend management."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GPUBackend._initialized:
            return
        GPUBackend._initialized = True
        self._gpu_enabled = True
        
        if not CUPY_AVAILABLE:
            print("[GPU] CuPy not available, using CPU backend")
            self._gpu_enabled = False
            return
        
        try:
            device = cp.cuda.Device()
            print(f"[GPU] Initialized: Device {device.id}, {device.mem_info[1] / (1024**3):.1f} GB VRAM")
            self._configure_memory_pool()
            self._warmup_kernels()
        except Exception as e:
            print(f"[GPU] Initialization failed: {e}")
            self._gpu_enabled = False
    
    def _configure_memory_pool(self):
        """Configure CuPy memory pool for optimal performance."""
        try:
            mempool = cp.get_default_memory_pool()
            free_mem = cp.cuda.Device().mem_info[0]
            mempool.set_limit(size=int(free_mem * 0.9))
            print(f"[GPU] Memory pool configured: {free_mem / (1024**3):.1f} GB limit")
        except Exception as e:
            print(f"[GPU] Memory pool config warning: {e}")
    
    def _warmup_kernels(self):
        """Pre-compile common CUDA kernels."""
        try:
            data = cp.zeros((16, 16, 16), dtype=cp.float32)
            mask = data > 0
            gpu_ndimage.binary_dilation(mask)
            gpu_ndimage.maximum_filter(data, size=3)
            gpu_ndimage.distance_transform_edt(mask.astype(cp.uint8))
            cp.argwhere(mask)
            del data, mask
            cp.get_default_memory_pool().free_all_blocks()
            print("[GPU] Kernel warmup completed")
        except Exception as e:
            print(f"[GPU] Kernel warmup failed (non-critical): {e}")
    
    @property
    def available(self) -> bool:
        """Check if GPU is available and enabled."""
        return CUPY_AVAILABLE and self._gpu_enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable GPU acceleration."""
        self._gpu_enabled = enabled
    
    def get_free_memory_mb(self) -> float:
        """Get free GPU memory in MB."""
        if not self.available:
            return 0.0
        try:
            free = cp.cuda.Device().mem_info[0]
            return free / (1024 * 1024)
        except Exception:
            return 0.0
    
    def can_fit(self, size_bytes: int, safety_factor: float = 0.8) -> bool:
        """Check if data can fit in GPU memory."""
        if not self.available:
            return False
        free_mb = self.get_free_memory_mb()
        required_mb = size_bytes / (1024 * 1024)
        return required_mb < (free_mb * safety_factor)
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Transfer NumPy array to GPU."""
        if self.available and isinstance(array, np.ndarray):
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Transfer GPU array to CPU."""
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def clear_memory(self, force: bool = False):
        """Clear GPU memory pool. Use force=True only when memory is critically low."""
        if CUPY_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            if force:
                mempool.free_all_blocks()
            else:
                mempool.free_all_free()
    
    def create_stream(self, non_blocking: bool = True) -> Any:
        """Create a CUDA stream for async operations."""
        return cp.cuda.Stream(non_blocking=non_blocking) if self.available else None
    
    def to_gpu_async(self, array: np.ndarray, stream: Any = None) -> Any:
        """Transfer NumPy array to GPU asynchronously."""
        if not self.available or not isinstance(array, np.ndarray):
            return array
        if stream:
            with stream:
                return cp.asarray(array)
        return cp.asarray(array)
    
    def to_cpu_async(self, array: Any, stream: Any = None) -> np.ndarray:
        """Transfer GPU array to CPU asynchronously."""
        if not CUPY_AVAILABLE or not isinstance(array, cp.ndarray):
            return array
        if stream:
            with stream:
                return cp.asnumpy(array)
        return cp.asnumpy(array)
    
    def synchronize_stream(self, stream: Any):
        """Wait for stream operations to complete."""
        if stream and self.available:
            stream.synchronize()


# Global singleton
_backend: Optional[GPUBackend] = None


def get_gpu_backend() -> GPUBackend:
    """Get the global GPU backend instance."""
    global _backend
    if _backend is None:
        _backend = GPUBackend()
    return _backend


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return get_gpu_backend().available
