"""
Smart DICOM loader with automatic strategy selection and performance optimizations.
Includes Numba JIT acceleration for pixel processing.
"""

import os
import re
import numpy as np
import pydicom
import concurrent.futures
from glob import glob
from typing import List, Tuple, Optional, Callable
from enum import Enum

from core import BaseLoader, VolumeData

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[SmartLoader] Numba not available, using standard numpy operations")


def _natural_sort_key(text: str):
    """Natural sorting key for filenames like img_1, img_2, ..., img_10"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


# Numba-accelerated functions (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _rescale_volume_numba(volume: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
        """Numba JIT accelerated volume rescaling with per-slice slopes/intercepts."""
        n_slices = volume.shape[0]
        for i in prange(n_slices):
            slope = slopes[i]
            intercept = intercepts[i]
            for j in range(volume.shape[1]):
                for k in range(volume.shape[2]):
                    volume[i, j, k] = volume[i, j, k] * slope + intercept
    
    @jit(nopython=True, parallel=True, cache=True)
    def _rescale_slice_numba(arr: np.ndarray, slope: float, intercept: float):
        """Numba JIT accelerated single slice rescaling."""
        for i in prange(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = arr[i, j] * slope + intercept


class LoadStrategy(Enum):
    """Loading strategy options."""
    FULL = "full"           # Load entire dataset
    FAST = "fast"           # Downsample for preview
    MEMORY_MAPPED = "mmap"  # Memory-mapped for very large
    CHUNKED = "chunked"     # Load in chunks


class SmartDicomLoader(BaseLoader):
    """
    Intelligent DICOM loader that automatically selects the best loading strategy
    based on dataset size and available system resources.
    
    Features:
    - Automatic strategy selection based on file count
    - Numba JIT acceleration (if available)
    - Parallel file reading
    - Memory-efficient processing
    - Progress callbacks with cancel support
    """
    
    # Thresholds for automatic strategy selection
    THRESHOLD_FAST = 200       # Use fast loader above this many slices
    THRESHOLD_MMAP = 500       # Use memory-mapped above this many slices
    THRESHOLD_CHUNKED = 1000   # Use chunked loader above this many slices
    
    def __init__(self, 
                 strategy: Optional[LoadStrategy] = None,
                 downsample_step: int = 2,
                 max_workers: int = 4,
                 use_numba: bool = True):
        """
        Args:
            strategy: Loading strategy. If None, auto-selects based on file count.
            downsample_step: Step size for fast loading (default: 2).
            max_workers: Number of parallel threads for file reading.
            use_numba: Whether to use Numba acceleration if available.
        """
        self.strategy = strategy
        self.downsample_step = downsample_step
        self.max_workers = max_workers
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self._cancelled = False
    
    def cancel(self):
        """Cancel ongoing loading operation."""
        self._cancelled = True
    
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
        self._cancelled = False
        
        print(f"[SmartLoader] Scanning: {folder_path}")
        if callback: callback(0, "Scanning directory...")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        # Find DICOM files
        files = self._find_dicom_files(folder_path)
        n_files = len(files)
        print(f"[SmartLoader] Found {n_files} DICOM files")
        
        # Auto-select strategy if not specified
        strategy = self._select_strategy(n_files)
        if callback: callback(5, f"Strategy: {strategy.value} ({n_files} files)")
        
        # Sort files by filename (natural sort)
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        # Load based on strategy
        if strategy == LoadStrategy.FAST:
            return self._load_fast(files, callback)
        elif strategy == LoadStrategy.MEMORY_MAPPED:
            return self._load_memory_mapped(files, callback)
        elif strategy == LoadStrategy.CHUNKED:
            return self._load_chunked(files, callback)
        else:
            return self._load_full(files, callback)
    
    def _select_strategy(self, n_files: int) -> LoadStrategy:
        """Auto-select loading strategy based on file count."""
        if self.strategy is not None:
            return self.strategy
        
        if n_files >= self.THRESHOLD_CHUNKED:
            print(f"[SmartLoader] Auto-selected CHUNKED strategy (>{self.THRESHOLD_CHUNKED} files)")
            return LoadStrategy.CHUNKED
        elif n_files >= self.THRESHOLD_MMAP:
            print(f"[SmartLoader] Auto-selected MEMORY_MAPPED strategy (>{self.THRESHOLD_MMAP} files)")
            return LoadStrategy.MEMORY_MAPPED
        elif n_files >= self.THRESHOLD_FAST:
            print(f"[SmartLoader] Auto-selected FAST strategy (>{self.THRESHOLD_FAST} files)")
            return LoadStrategy.FAST
        else:
            print(f"[SmartLoader] Auto-selected FULL strategy (<{self.THRESHOLD_FAST} files)")
            return LoadStrategy.FULL
    
    def _find_dicom_files(self, folder_path: str) -> List[str]:
        """Find DICOM files efficiently."""
        files = glob(os.path.join(folder_path, "*.dcm"))
        if not files:
            # Check all files for DICOM header
            all_files = [f for f in glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
            files = []
            for f in all_files:
                try:
                    pydicom.dcmread(f, stop_before_pixels=True)
                    files.append(f)
                except:
                    continue
        if not files:
            raise FileNotFoundError("No valid DICOM files found")
        return files
    
    def _read_files_parallel(self, files: List[str], 
                              callback: Optional[Callable] = None,
                              start_percent: int = 10,
                              end_percent: int = 50) -> List[pydicom.dataset.FileDataset]:
        """Read DICOM files in parallel, maintaining order."""
        def read_single(args):
            idx, f = args
            try:
                return (idx, pydicom.dcmread(f))
            except Exception as e:
                print(f"Warning: Failed to read {f} - {e}")
                return (idx, None)
        
        total = len(files)
        results = [None] * total
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(read_single, (i, f)): i for i, f in enumerate(files)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                if self._cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise InterruptedError("Loading cancelled by user")
                
                idx, ds = future.result()
                results[idx] = ds
                completed += 1
                
                if callback and completed % 20 == 0:
                    percent = start_percent + int((end_percent - start_percent) * completed / total)
                    callback(percent, f"Reading slice {completed}/{total}...")
        
        return [r for r in results if r is not None]
    
    def _sort_slices_by_position(self, slices: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """Sort slices by ImagePositionPatient Z coordinate."""
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
            try:
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
                z_start = float(slices[0].ImagePositionPatient[2])
                z_end = float(slices[-1].ImagePositionPatient[2])
                print(f"[SmartLoader] Sorted by Z-position: {z_start:.2f} -> {z_end:.2f}")
            except Exception as e:
                print(f"[SmartLoader] Warning: Could not sort by position: {e}")
        return slices
    
    def _build_volume_optimized(self, slices: List[pydicom.dataset.FileDataset],
                                 callback: Optional[Callable] = None,
                                 downsample: int = 1) -> Tuple[np.ndarray, tuple, tuple]:
        """Build volume with optimized memory operations and optional Numba acceleration."""
        first = slices[0]
        pixel_spacing = first.PixelSpacing
        
        # Calculate Z spacing
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
            z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        else:
            z_spacing = getattr(first, 'SliceThickness', 1.0)
        
        if downsample > 1:
            spacing = (float(pixel_spacing[0]) * downsample, 
                      float(pixel_spacing[1]) * downsample, 
                      float(z_spacing))
            base_shape = first.pixel_array.shape
            rows = base_shape[0] // downsample
            cols = base_shape[1] // downsample
        else:
            spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
            rows, cols = first.pixel_array.shape
        
        origin = tuple(getattr(first, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
        
        # Pre-allocate volume
        n_slices = len(slices)
        volume = np.empty((n_slices, rows, cols), dtype=np.float32)
        
        # Collect rescale parameters
        slopes = np.empty(n_slices, dtype=np.float32)
        intercepts = np.empty(n_slices, dtype=np.float32)
        
        # First pass: copy pixel data and collect rescale params
        if callback: callback(60, "Copying pixel data...")
        
        for i, s in enumerate(slices):
            slopes[i] = float(getattr(s, 'RescaleSlope', 1.0))
            intercepts[i] = float(getattr(s, 'RescaleIntercept', 0.0))
            
            if downsample > 1:
                arr = s.pixel_array[::downsample, ::downsample]
                volume[i] = arr[:rows, :cols].astype(np.float32)
            else:
                volume[i] = s.pixel_array.astype(np.float32)
            
            if callback and i % 50 == 0:
                percent = 60 + int(30 * i / n_slices)
                callback(percent, f"Processing slice {i+1}/{n_slices}...")
        
        # Apply rescaling (use Numba if available)
        if callback: callback(90, "Applying rescale (Numba)..." if self.use_numba else "Applying rescale...")
        
        if self.use_numba and NUMBA_AVAILABLE:
            _rescale_volume_numba(volume, slopes, intercepts)
        else:
            # Fallback to vectorized numpy
            for i in range(n_slices):
                if slopes[i] != 1.0:
                    volume[i] *= slopes[i]
                if intercepts[i] != 0.0:
                    volume[i] += intercepts[i]
        
        return volume, spacing, origin
    
    def _load_full(self, files: List[str], callback: Optional[Callable] = None) -> VolumeData:
        """Full resolution loading."""
        if callback: callback(10, f"Reading {len(files)} slices (full)...")
        
        slices = self._read_files_parallel(files, callback, 10, 50)
        if not slices:
            raise ValueError("No valid DICOM slices loaded")
        
        slices = self._sort_slices_by_position(slices)
        
        if callback: callback(55, "Building volume...")
        volume, spacing, origin = self._build_volume_optimized(slices, callback)
        
        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "LoadStrategy": "Full",
            "NumbaAccelerated": self.use_numba and NUMBA_AVAILABLE
        }
        
        if callback: callback(100, "Loading complete.")
        print(f"[SmartLoader] Complete: {volume.shape}, Spacing: {spacing}")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)
    
    def _load_fast(self, files: List[str], callback: Optional[Callable] = None) -> VolumeData:
        """Fast loading with downsampling."""
        step = self.downsample_step
        selected_files = files[::step]
        
        if callback: callback(10, f"Fast loading: {len(selected_files)}/{len(files)} slices...")
        
        slices = self._read_files_parallel(selected_files, callback, 10, 50)
        if not slices:
            raise ValueError("No valid DICOM slices loaded")
        
        slices = self._sort_slices_by_position(slices)
        
        if callback: callback(55, "Building downsampled volume...")
        volume, spacing, origin = self._build_volume_optimized(slices, callback, downsample=step)
        
        # Adjust Z spacing for skipped slices
        spacing = (spacing[0], spacing[1], spacing[2] * step if len(files) > len(selected_files) else spacing[2])
        
        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "LoadStrategy": f"Fast (step={step})",
            "OriginalSliceCount": len(files),
            "NumbaAccelerated": self.use_numba and NUMBA_AVAILABLE
        }
        
        if callback: callback(100, "Fast loading complete.")
        print(f"[SmartLoader] Fast complete: {volume.shape}")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)
    
    def _load_memory_mapped(self, files: List[str], callback: Optional[Callable] = None) -> VolumeData:
        """Memory-mapped loading for very large datasets."""
        import tempfile
        
        if callback: callback(10, "Reading metadata...")
        
        # Read first file for metadata
        first_ds = pydicom.dcmread(files[0])
        rows, cols = first_ds.pixel_array.shape
        n_slices = len(files)
        
        pixel_spacing = first_ds.PixelSpacing
        z_spacing = getattr(first_ds, 'SliceThickness', 1.0)
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(getattr(first_ds, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
        
        # Create memory-mapped file
        mmap_file = os.path.join(tempfile.gettempdir(), f"ct_mmap_{id(self)}_{n_slices}.dat")
        if callback: callback(15, f"Creating memory map: {n_slices}x{rows}x{cols}...")
        
        volume = np.memmap(mmap_file, dtype=np.float32, mode='w+', shape=(n_slices, rows, cols))
        
        # Read and process slices
        for i, f in enumerate(files):
            if self._cancelled:
                raise InterruptedError("Loading cancelled")
            
            ds = pydicom.dcmread(f)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            
            arr = ds.pixel_array.astype(np.float32)
            if slope != 1.0:
                arr *= slope
            if intercept != 0.0:
                arr += intercept
            volume[i] = arr
            
            if callback and i % 20 == 0:
                percent = 15 + int(80 * i / n_slices)
                callback(percent, f"Mapping slice {i+1}/{n_slices}...")
        
        volume.flush()
        
        metadata = {
            "SampleID": getattr(first_ds, "PatientID", "Unknown"),
            "ScanType": getattr(first_ds, "Modality", "CT"),
            "SliceCount": n_slices,
            "LoadStrategy": "MemoryMapped",
            "MmapFile": mmap_file
        }
        
        if callback: callback(100, "Memory-mapped loading complete.")
        print(f"[SmartLoader] MMap complete: {volume.shape}")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)
    
    def _load_chunked(self, files: List[str], callback: Optional[Callable] = None) -> VolumeData:
        """Chunked loading - loads only first chunk, rest on demand."""
        chunk_size = 64
        
        if callback: callback(10, "Reading metadata...")
        
        # Read first file for metadata
        first_ds = pydicom.dcmread(files[0])
        rows, cols = first_ds.pixel_array.shape
        n_slices = len(files)
        
        pixel_spacing = first_ds.PixelSpacing
        z_spacing = getattr(first_ds, 'SliceThickness', 1.0)
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(getattr(first_ds, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
        
        # Load only first chunk for preview
        preview_slices = min(chunk_size, n_slices)
        if callback: callback(20, f"Loading preview chunk (0-{preview_slices})...")
        
        preview_files = files[:preview_slices]
        slices = self._read_files_parallel(preview_files, callback, 20, 80)
        slices = self._sort_slices_by_position(slices)
        
        volume, _, _ = self._build_volume_optimized(slices, callback)
        
        metadata = {
            "SampleID": getattr(first_ds, "PatientID", "Unknown"),
            "ScanType": getattr(first_ds, "Modality", "CT"),
            "SliceCount": preview_slices,
            "TotalSliceCount": n_slices,
            "LoadStrategy": "Chunked (preview)",
            "ChunkSize": chunk_size,
            "FullDimensions": (n_slices, rows, cols)
        }
        
        if callback: callback(100, f"Chunked loading complete ({preview_slices}/{n_slices} slices).")
        print(f"[SmartLoader] Chunked preview: {volume.shape} ({preview_slices}/{n_slices})")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)
