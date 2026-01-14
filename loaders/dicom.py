"""
DICOM series loaders for CT/Micro-CT data.
Optimized for uncompressed files with parallel reading and filename-based sorting.
Includes Numba JIT acceleration for pixel processing when available.
"""

import os
import re
import gc
import numpy as np
import pydicom
import concurrent.futures
from glob import glob
from typing import List, Tuple, Optional, Callable
from enum import Enum

from core import BaseLoader, VolumeData
from config import (
    LOADER_THRESHOLD_FAST,
    LOADER_THRESHOLD_MMAP,
    LOADER_THRESHOLD_CHUNKED,
    LOADER_MAX_WORKERS,
    LOADER_DOWNSAMPLE_STEP
)

# Try to import Numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Numba-accelerated rescale (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _rescale_volume_numba(volume: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
        """Numba JIT accelerated volume rescaling."""
        n_slices = volume.shape[0]
        for i in prange(n_slices):
            slope = slopes[i]
            intercept = intercepts[i]
            if slope != 1.0 or intercept != 0.0:
                for j in range(volume.shape[1]):
                    for k in range(volume.shape[2]):
                        volume[i, j, k] = volume[i, j, k] * slope + intercept
else:
    def _rescale_volume_numba(volume, slopes, intercepts):
        """Fallback: vectorized numpy rescale."""
        for i in range(len(slopes)):
            if slopes[i] != 1.0:
                volume[i] *= slopes[i]
            if intercepts[i] != 0.0:
                volume[i] += intercepts[i]


def _natural_sort_key(text: str):
    """Natural sorting key for filenames like img_1, img_2, ..., img_10"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


# ==========================================
# Shared Utility Functions
# ==========================================

def _validate_path(folder_path: str) -> None:
    """Validate folder path exists."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Path does not exist: {folder_path}")


def _calculate_z_spacing(ds1, ds2=None, step: int = 1) -> float:
    """
    Calculate Z-spacing from DICOM headers.
    
    Args:
        ds1: First DICOM dataset
        ds2: Second DICOM dataset (optional, for calculating from position difference)
        step: Multiplier for spacing (used by downsampling loaders)
    """
    if ds2 is not None:
        if hasattr(ds1, 'ImagePositionPatient') and hasattr(ds2, 'ImagePositionPatient'):
            return abs(float(ds2.ImagePositionPatient[2]) - float(ds1.ImagePositionPatient[2]))
    return float(getattr(ds1, 'SliceThickness', 1.0)) * step


def _get_spacing_and_origin(ds1, ds2=None, xy_step: int = 1, z_step: int = 1) -> Tuple[tuple, tuple]:
    """
    Extract spacing and origin from DICOM dataset.
    
    Args:
        ds1: First DICOM dataset
        ds2: Second DICOM dataset (for Z-spacing calculation)
        xy_step: Multiplier for XY spacing (used by downsampling)
        z_step: Multiplier for Z spacing (used by downsampling)
    """
    pixel_spacing = ds1.PixelSpacing
    z_spacing = _calculate_z_spacing(ds1, ds2, z_step)
    spacing = (
        float(pixel_spacing[0]) * xy_step,
        float(pixel_spacing[1]) * xy_step,
        z_spacing
    )
    origin = tuple(getattr(ds1, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
    return spacing, origin


def _apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
    """
    Apply DICOM rescale slope and intercept to pixel array (in-place).
    
    Args:
        arr: Pixel array (float32)
        ds: DICOM dataset with RescaleSlope/RescaleIntercept
    """
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    if slope != 1.0:
        arr *= slope
    if intercept != 0.0:
        arr += intercept
    return arr


def _sort_slices_by_position(slices: List, loader_name: str = "Loader") -> List:
    """
    Sort DICOM slices by ImagePositionPatient Z-coordinate.
    
    Args:
        slices: List of DICOM datasets
        loader_name: Name for logging
    """
    if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            z_start = float(slices[0].ImagePositionPatient[2])
            z_end = float(slices[-1].ImagePositionPatient[2])
            print(f"[{loader_name}] Sorted by Z-position: {z_start:.2f} -> {z_end:.2f}")
        except Exception as e:
            print(f"[{loader_name}] Warning: Could not sort by header position: {e}")
    return slices


def _extract_metadata(ds, slice_count: int, extra: dict = None) -> dict:
    """
    Extract common metadata from DICOM dataset.
    
    Args:
        ds: DICOM dataset
        slice_count: Number of slices
        extra: Additional metadata to include
    """
    metadata = {
        "SampleID": getattr(ds, "PatientID", "Unknown Sample"),
        "ScanType": getattr(ds, "Modality", "CT"),
        "SliceCount": slice_count,
        "SortMethod": "Header" if hasattr(ds, 'ImagePositionPatient') else "Filename"
    }
    if extra:
        metadata.update(extra)
    return metadata


def _find_dicom_files(folder_path: str) -> List[str]:
    """Find DICOM files in folder, checking extension first then content."""
    files = glob(os.path.join(folder_path, "*.dcm"))
    if not files:
        # Check all files for DICOM magic bytes or valid header
        all_files = glob(os.path.join(folder_path, "*"))
        files = []
        for f in all_files:
            if os.path.isfile(f):
                try:
                    pydicom.dcmread(f, stop_before_pixels=True)
                    files.append(f)
                except:
                    continue
    if not files:
        raise FileNotFoundError("No valid DICOM/CT files found")
    return files


class DicomSeriesLoader(BaseLoader):
    """Concrete DICOM series loader for Industrial CT/Micro-CT scans.
    
    Optimized for performance with:
    - Filename-based natural sorting (default, fast and reliable)
    - Optional header-based sorting verification
    - Parallel file reading using ThreadPoolExecutor
    """

    def __init__(self, use_header_sort: bool = True, max_workers: int = 4):
        """
        Args:
            use_header_sort: If True, sort files using ImagePositionPatient headers (default).
                            This ensures correct Z-axis ordering regardless of filename.
                            Falls back to filename sort if headers are missing/corrupt.
            max_workers: Number of parallel threads for file reading.
        """
        self.use_header_sort = use_header_sort
        self.max_workers = max_workers

    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Scanning sample folder: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")

        _validate_path(folder_path)

        files = _find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Sorting...")
        
        # Sort files by filename (natural sort)
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        # Optionally verify with header positions
        if self.use_header_sort:
            files = self._verify_sort_with_headers(files, callback)
        
        if callback: callback(20, f"Reading {len(files)} slices...")
        slices = self._parallel_read_files(files, callback)
        
        if not slices:
            raise ValueError("No valid DICOM slices loaded.")
        
        # Sort slices by Z-position
        slices = _sort_slices_by_position(slices, "Loader")
        
        if callback: callback(50, "Building 3D volume...")
        volume, spacing, origin = self._build_volume(slices, callback)

        metadata = _extract_metadata(slices[0], len(slices), {
            "Description": getattr(slices[0], "StudyDescription", "No Description")
        })

        print(f"[Loader] Loading complete: {volume.shape}, Voxel Spacing: {spacing}")
        if callback: callback(100, "Loading complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)



    def _verify_sort_with_headers(self, files: List[str], 
                                   callback: Optional[Callable] = None) -> List[str]:
        """Optionally verify sort order using ImagePositionPatient headers."""
        try:
            file_positions = []
            total = len(files)
            for i, f in enumerate(files):
                if callback and i % 50 == 0:
                    callback(10 + int(10 * i / total), f"Reading header {i+1}/{total}...")
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    file_positions.append((f, float(ds.ImagePositionPatient[2])))
                else:
                    print(f"[Loader] Header missing ImagePositionPatient, using filename order")
                    return files
            
            # Sort by Z position
            file_positions.sort(key=lambda x: x[1])
            print(f"[Loader] Header sort verified. Z-range: {file_positions[0][1]:.2f} -> {file_positions[-1][1]:.2f}")
            return [fp[0] for fp in file_positions]
        except Exception as e:
            print(f"[Loader] Header sort failed ({e}), using filename order")
            return files

    def _parallel_read_files(self, files: List[str], 
                             callback: Optional[Callable] = None) -> List[pydicom.dataset.FileDataset]:
        """Parallel DICOM file reading using ThreadPoolExecutor."""
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
                idx, ds = future.result()
                results[idx] = ds
                completed += 1
                if callback and completed % 20 == 0:
                    percent = 20 + int(30 * completed / total)
                    callback(percent, f"Reading slice {completed}/{total}...")
        
        return [r for r in results if r is not None]

    def _build_volume(self, slices: List[pydicom.dataset.FileDataset], 
                      callback: Optional[Callable] = None) -> Tuple[np.ndarray, tuple, tuple]:
        """Build 3D volume from DICOM slices with optimized memory operations."""
        ds2 = slices[1] if len(slices) > 1 else None
        spacing, origin = _get_spacing_and_origin(slices[0], ds2)

        img_shape = list(slices[0].pixel_array.shape)
        img_shape.insert(0, len(slices))
        
        # Pre-allocate with np.empty (faster than zeros)
        volume = np.empty(img_shape, dtype=np.float32)
        
        # Collect rescale parameters for batch processing
        total = len(slices)
        slopes = np.empty(total, dtype=np.float32)
        intercepts = np.empty(total, dtype=np.float32)

        # First pass: copy pixel data and collect rescale params
        for i, s in enumerate(slices):
            if callback and i % 20 == 0:
                percent = 50 + int(40 * i / total)
                callback(percent, f"Reading slice {i+1}/{total}...")
            
            slopes[i] = float(getattr(s, 'RescaleSlope', 1.0))
            intercepts[i] = float(getattr(s, 'RescaleIntercept', 0.0))
            volume[i] = s.pixel_array.astype(np.float32)
        
        # Second pass: apply rescale (Numba accelerated if available)
        if callback: callback(92, "Applying rescale (Numba)..." if NUMBA_AVAILABLE else "Applying rescale...")
        _rescale_volume_numba(volume, slopes, intercepts)
        
        if callback: callback(98, "Finalizing...")
        gc.collect()

        return volume, spacing, origin


class FastDicomLoader(DicomSeriesLoader):
    """
    Fast loader that downsamples data (lowers resolution).
    Essential for previewing large Micro-CT datasets.
    """

    def __init__(self, step: int = 2, max_workers: int = 4):
        super().__init__(use_header_sort=False, max_workers=max_workers)
        self.step = step

    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[FastLoader] Scanning (Step={self.step}): {folder_path} ...")
        if callback: callback(0, f"Fast scanning (step={self.step})...")

        _validate_path(folder_path)
        files = _find_dicom_files(folder_path)

        print("[FastLoader] Sorting files by natural filename order...")
        if callback: callback(10, "Sorting files...")
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))

        selected_files = files[::self.step]
        print(f"[FastLoader] Selected {len(selected_files)} / {len(files)} files.")

        if callback: callback(20, f"Reading {len(selected_files)} slices...")
        slices = self._parallel_read_files(selected_files, callback)
        
        if not slices:
            raise ValueError("No valid slices loaded.")
        
        slices = _sort_slices_by_position(slices, "FastLoader")

        if callback: callback(50, "Building downsampled volume...")
        volume, spacing, origin = self._build_volume_downsampled(slices, callback)

        metadata = _extract_metadata(slices[0], len(slices), {"Type": "Fast/Downsampled"})

        print(f"[FastLoader] Complete: {volume.shape}, Spacing: {spacing}")
        if callback: callback(100, "Fast load complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _build_volume_downsampled(self, slices: List[pydicom.dataset.FileDataset], 
                                   callback: Optional[Callable] = None) -> Tuple[np.ndarray, tuple, tuple]:
        ds2 = slices[1] if len(slices) > 1 else None
        spacing, origin = _get_spacing_and_origin(slices[0], ds2, xy_step=self.step, z_step=self.step)

        base_shape = slices[0].pixel_array.shape
        new_h = base_shape[0] // self.step
        new_w = base_shape[1] // self.step

        volume = np.empty((len(slices), new_h, new_w), dtype=np.float32)

        total = len(slices)
        for i, s in enumerate(slices):
            if callback and i % 10 == 0:
                percent = 50 + int(50 * i / total)
                callback(percent, f"Downsampling slice {i+1}/{total}...")
                
            arr = s.pixel_array[::self.step, ::self.step].astype(np.float32)
            arr = arr[:new_h, :new_w]
            _apply_rescale(arr, s)
            volume[i] = arr

        return volume, spacing, origin


class MemoryMappedDicomLoader(DicomSeriesLoader):
    """
    Memory-mapped loader for very large DICOM datasets.
    Uses numpy memory-mapping to avoid loading entire volume into RAM.
    """
    
    def __init__(self, cache_dir: str = None, use_header_sort: bool = False, max_workers: int = 4):
        super().__init__(use_header_sort=use_header_sort, max_workers=max_workers)
        import tempfile
        self.cache_dir = cache_dir or tempfile.gettempdir()
    
    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[MemoryMappedLoader] Scanning: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")
        
        _validate_path(folder_path)
        files = _find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Sorting...")
        
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        if self.use_header_sort:
            files = self._verify_sort_with_headers(files, callback)
        
        print(f"[MemoryMappedLoader] Found {len(files)} valid slices")
        
        # Read first and second file for metadata
        first_full = pydicom.dcmread(files[0])
        second_ds = pydicom.dcmread(files[1], stop_before_pixels=True) if len(files) > 1 else None
        rows, cols = first_full.pixel_array.shape
        num_slices = len(files)
        
        spacing, origin = _get_spacing_and_origin(first_full, second_ds)
        
        mmap_file = os.path.join(self.cache_dir, f"ct_volume_{id(self)}_{num_slices}.dat")
        print(f"[MemoryMappedLoader] Creating memory-mapped file: {mmap_file}")
        if callback: callback(30, "Creating memory map...")
        
        volume = np.memmap(mmap_file, dtype=np.float32, mode='w+', shape=(num_slices, rows, cols))
        
        print(f"[MemoryMappedLoader] Loading {num_slices} slices into memory-mapped array...")
        for i, f in enumerate(files):
            if i % 10 == 0 and callback:
                percent = 30 + int(70 * i / num_slices)
                callback(percent, f"Mapping slice {i+1}/{num_slices}...")
            
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
            _apply_rescale(arr, ds)
            volume[i] = arr
        
        volume.flush()
        print(f"[MemoryMappedLoader] Complete: {volume.shape}, Memory-mapped to disk")
        
        metadata = _extract_metadata(first_full, num_slices, {
            "Type": "Memory-Mapped",
            "MmapFile": mmap_file
        })
        
        if callback: callback(100, "Memory map complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)


class ChunkedDicomLoader(DicomSeriesLoader):
    """
    Chunked loader that loads data in smaller chunks for memory efficiency.
    """
    
    def __init__(self, chunk_size: int = 64, use_header_sort: bool = False):
        super().__init__(use_header_sort=use_header_sort, max_workers=4)
        self.chunk_size = chunk_size
        self._file_list = None
        self._metadata = None
        self._current_chunk = None
        self._current_chunk_range = None
    
    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[ChunkedLoader] Preparing chunked access: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")
        
        _validate_path(folder_path)
        files = _find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Sorting...")
        
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        if self.use_header_sort:
            files = self._verify_sort_with_headers(files, callback)
        
        self._file_list = files
        
        if callback: callback(50, "Reading first slice metadata...")
        first = pydicom.dcmread(self._file_list[0])
        second = pydicom.dcmread(self._file_list[1], stop_before_pixels=True) if len(files) > 1 else None
        rows, cols = first.pixel_array.shape
        num_slices = len(self._file_list)
        
        spacing, origin = _get_spacing_and_origin(first, second)
        
        self._metadata = _extract_metadata(first, num_slices, {
            "Type": "Chunked",
            "ChunkSize": self.chunk_size,
            "Dimensions": (num_slices, rows, cols)
        })
        
        print(f"[ChunkedLoader] Loading first chunk (0-{min(self.chunk_size, num_slices)})...")
        if callback: callback(80, "Loading initial preview chunk...")
        chunk = self.load_chunk(0)
        
        print(f"[ChunkedLoader] Ready. Total slices: {num_slices}, Chunk size: {self.chunk_size}")
        
        if callback: callback(100, "Initialization complete.")
        return VolumeData(raw_data=chunk, spacing=spacing, origin=origin, metadata=self._metadata)
    
    def load_chunk(self, start_slice: int) -> np.ndarray:
        if self._file_list is None:
            raise RuntimeError("Must call load() first to initialize the loader")
        
        end_slice = min(start_slice + self.chunk_size, len(self._file_list))
        
        if self._current_chunk_range == (start_slice, end_slice):
            return self._current_chunk
        
        print(f"[ChunkedLoader] Loading chunk: slices {start_slice}-{end_slice}")
        
        first = pydicom.dcmread(self._file_list[0])
        rows, cols = first.pixel_array.shape
        chunk_size = end_slice - start_slice
        
        chunk = np.empty((chunk_size, rows, cols), dtype=np.float32)
        
        for i, idx in enumerate(range(start_slice, end_slice)):
            ds = pydicom.dcmread(self._file_list[idx])
            arr = ds.pixel_array.astype(np.float32)
            _apply_rescale(arr, ds)
            chunk[i] = arr
        
        self._current_chunk = chunk
        self._current_chunk_range = (start_slice, end_slice)
        
        return chunk
    
    def get_total_slices(self) -> int:
        return len(self._file_list) if self._file_list else 0
    
    def get_num_chunks(self) -> int:
        total = self.get_total_slices()
        return (total + self.chunk_size - 1) // self.chunk_size


# ==========================================
# Strategy Enum and Smart Loader
# ==========================================

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
    
    Delegates to specialized loaders:
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
