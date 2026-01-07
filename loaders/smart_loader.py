"""
Smart DICOM loader with automatic strategy selection.
Delegates to existing dicom.py loaders based on dataset size.
"""

import os
from glob import glob
from typing import Optional, Callable
from enum import Enum

from core import BaseLoader, VolumeData
from loaders.dicom import (
    DicomSeriesLoader,
    FastDicomLoader, 
    MemoryMappedDicomLoader,
    ChunkedDicomLoader
)
from config import (
    LOADER_THRESHOLD_FAST,
    LOADER_THRESHOLD_MMAP,
    LOADER_THRESHOLD_CHUNKED,
    LOADER_MAX_WORKERS,
    LOADER_DOWNSAMPLE_STEP
)


class LoadStrategy(Enum):
    """Loading strategy options."""
    AUTO = "auto"           # Automatically select best strategy
    FULL = "full"           # Full resolution loading
    FAST = "fast"           # Downsampled preview
    MEMORY_MAPPED = "mmap"  # Memory-mapped for very large files
    CHUNKED = "chunked"     # Chunked loading for huge files


class SmartDicomLoader(BaseLoader):
    """
    Intelligent DICOM loader that automatically selects the best loading strategy
    based on dataset size and available system resources.
    
    This is a facade/wrapper that delegates to specialized loaders in dicom.py:
    - DicomSeriesLoader: Full resolution loading
    - FastDicomLoader: Downsampled preview
    - MemoryMappedDicomLoader: Memory-mapped for very large files
    - ChunkedDicomLoader: Chunked loading for huge files
    
    Usage:
        # Auto-select strategy
        loader = SmartDicomLoader()
        data = loader.load("path/to/dicom")
        
        # Force specific strategy
        loader = SmartDicomLoader(strategy=LoadStrategy.FAST)
        data = loader.load("path/to/dicom")
    """

    def __init__(self, 
                 strategy: Optional[LoadStrategy] = None,
                 downsample_step: int = LOADER_DOWNSAMPLE_STEP,
                 max_workers: int = LOADER_MAX_WORKERS):
        """
        Args:
            strategy: Loading strategy. If None or AUTO, auto-selects based on file count.
            downsample_step: Step size for fast loading (default: 2).
            max_workers: Number of parallel threads for file reading.
        """
        self.strategy = strategy
        self.downsample_step = downsample_step
        self.max_workers = max_workers
        self._selected_loader: Optional[BaseLoader] = None
    
    def load(self, folder_path: str, 
             callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        """
        Load DICOM series with automatic optimization.
        
        Args:
            folder_path: Path to the folder containing DICOM files.
            callback: Optional progress callback (percent, message).
            
        Returns:
            VolumeData with loaded volume.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        # Count files to determine strategy
        n_files = self._count_dicom_files(folder_path)
        print(f"[SmartLoader] Found {n_files} DICOM files")
        
        # Select strategy
        strategy = self._select_strategy(n_files)
        if callback:
            callback(5, f"Strategy: {strategy.value} ({n_files} files)")
        
        # Create appropriate loader
        self._selected_loader = self._create_loader(strategy)
        
        # Delegate to selected loader
        print(f"[SmartLoader] Using {type(self._selected_loader).__name__}")
        data = self._selected_loader.load(folder_path, callback=callback)
        
        # Add strategy info to metadata
        if data.metadata:
            data.metadata['LoadStrategy'] = strategy.value
            data.metadata['LoaderClass'] = type(self._selected_loader).__name__
        
        return data
    
    def _count_dicom_files(self, folder_path: str) -> int:
        """Count DICOM files in folder."""
        dcm_files = glob(os.path.join(folder_path, "*.dcm"))
        if dcm_files:
            return len(dcm_files)
        
        # Count all files (may be DICOM without extension)
        all_files = [f for f in glob(os.path.join(folder_path, "*")) 
                     if os.path.isfile(f)]
        return len(all_files)
    
    def _select_strategy(self, n_files: int) -> LoadStrategy:
        """Auto-select loading strategy based on file count."""
        # Use explicit strategy if provided
        if self.strategy is not None and self.strategy != LoadStrategy.AUTO:
            print(f"[SmartLoader] Using explicit strategy: {self.strategy.value}")
            return self.strategy
        
        # Auto-select based on file count
        if n_files >= LOADER_THRESHOLD_CHUNKED:
            print(f"[SmartLoader] Auto: CHUNKED (>{LOADER_THRESHOLD_CHUNKED} files)")
            return LoadStrategy.CHUNKED
        elif n_files >= LOADER_THRESHOLD_MMAP:
            print(f"[SmartLoader] Auto: MEMORY_MAPPED (>{LOADER_THRESHOLD_MMAP} files)")
            return LoadStrategy.MEMORY_MAPPED
        elif n_files >= LOADER_THRESHOLD_FAST:
            print(f"[SmartLoader] Auto: FAST (>{LOADER_THRESHOLD_FAST} files)")
            return LoadStrategy.FAST
        else:
            print(f"[SmartLoader] Auto: FULL (<{LOADER_THRESHOLD_FAST} files)")
            return LoadStrategy.FULL
    
    def _create_loader(self, strategy: LoadStrategy) -> BaseLoader:
        """Create the appropriate loader for the selected strategy."""
        if strategy == LoadStrategy.FAST:
            return FastDicomLoader(step=self.downsample_step, max_workers=self.max_workers)
        elif strategy == LoadStrategy.MEMORY_MAPPED:
            return MemoryMappedDicomLoader(max_workers=self.max_workers)
        elif strategy == LoadStrategy.CHUNKED:
            return ChunkedDicomLoader()
        else:  # FULL or AUTO
            return DicomSeriesLoader(max_workers=self.max_workers)
    
    @property
    def selected_loader(self) -> Optional[BaseLoader]:
        """Get the loader that was selected for the last load operation."""
        return self._selected_loader
