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
    """
    Singleton for GPU backend management.
    Handles GPU detection, memory queries, and array transfers.
    """
    
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
        
        if CUPY_AVAILABLE:
            try:
                device = cp.cuda.Device()
                total_mem = device.mem_info[1] / (1024**3)  # GB
                print(f"[GPU] Initialized: Device {device.id}, {total_mem:.1f} GB VRAM")
                
                # Kernel warmup: pre-compile common CUDA kernels to avoid first-call latency
                self._warmup_kernels()
            except Exception as e:
                print(f"[GPU] Initialization failed: {e}")
                self._gpu_enabled = False
        else:
            print("[GPU] CuPy not available, using CPU backend")
            self._gpu_enabled = False
    
    def _warmup_kernels(self):
        """Pre-compile common CUDA kernels with small dummy data."""
        try:
            # Small 16x16x16 warmup data
            warmup_data = cp.zeros((16, 16, 16), dtype=cp.float32)
            warmup_bool = warmup_data > 0
            
            # Warmup common ndimage operations
            gpu_ndimage.binary_dilation(warmup_bool)
            gpu_ndimage.maximum_filter(warmup_data, size=3)
            gpu_ndimage.distance_transform_edt(warmup_bool.astype(cp.uint8))
            
            # Warmup basic operations
            _ = cp.argwhere(warmup_bool)
            _ = cp.roll(warmup_data, 1, axis=0)
            
            # Cleanup
            del warmup_data, warmup_bool
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
    
    def clear_memory(self):
        """Clear GPU memory pool (use sparingly)."""
        if CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    def create_stream(self, non_blocking: bool = True) -> Any:
        """
        Create a CUDA stream for async operations.
        
        Args:
            non_blocking: If True, the stream can run concurrently with default stream
            
        Returns:
            CuPy CUDA Stream object, or None if GPU not available
        """
        if not self.available:
            return None
        return cp.cuda.Stream(non_blocking=non_blocking)
    
    def to_gpu_async(self, array: np.ndarray, stream: Any = None) -> Any:
        """
        Transfer NumPy array to GPU asynchronously.
        
        For best performance with async transfers, use pinned (page-locked) memory
        on the CPU side. This method works with regular numpy arrays but pinned
        memory will provide better PCIe throughput.
        
        Args:
            array: NumPy array to transfer
            stream: Optional CUDA stream for async transfer
            
        Returns:
            CuPy array on GPU
        """
        if not self.available or not isinstance(array, np.ndarray):
            return array
        
        if stream is not None:
            with stream:
                return cp.asarray(array)
        return cp.asarray(array)
    
    def to_cpu_async(self, array: Any, stream: Any = None) -> np.ndarray:
        """
        Transfer GPU array to CPU asynchronously.
        
        Args:
            array: CuPy array to transfer
            stream: Optional CUDA stream for async transfer
            
        Returns:
            NumPy array on CPU
        """
        if not CUPY_AVAILABLE or not isinstance(array, cp.ndarray):
            return array
        
        if stream is not None:
            with stream:
                return cp.asnumpy(array)
        return cp.asnumpy(array)
    
    def synchronize_stream(self, stream: Any):
        """Wait for all operations in the stream to complete."""
        if stream is not None and self.available:
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
