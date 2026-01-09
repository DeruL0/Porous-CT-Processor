"""
Core module containing base classes and data structures.
"""

from core.base import VolumeData, BaseLoader, BaseProcessor, BaseVisualizer
from core.gpu_backend import get_gpu_backend, is_gpu_available

__all__ = ['VolumeData', 'BaseLoader', 'BaseProcessor', 'BaseVisualizer', 
           'get_gpu_backend', 'is_gpu_available']
