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

from core import BaseLoader, VolumeData

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

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)
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
        
        # Sort slices by ImagePositionPatient if available (ensures correct Z-order)
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
            try:
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
                print(f"[Loader] Sorted by ImagePositionPatient Z: {slices[0].ImagePositionPatient[2]:.2f} -> {slices[-1].ImagePositionPatient[2]:.2f}")
            except Exception as e:
                print(f"[Loader] Warning: Could not sort by header position: {e}")
        
        if callback: callback(50, "Building 3D volume...")
        volume, spacing, origin = self._build_volume(slices, callback)

        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Description": getattr(slices[0], "StudyDescription", "No Description"),
            "SortMethod": "Header (ImagePositionPatient)" if hasattr(slices[0], 'ImagePositionPatient') else "Filename"
        }

        print(f"[Loader] Loading complete: {volume.shape}, Voxel Spacing: {spacing}")
        if callback: callback(100, "Loading complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _find_dicom_files(self, folder_path: str) -> List[str]:
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
        pixel_spacing = slices[0].PixelSpacing
        if len(slices) > 1:
            # Use header positions if available, otherwise estimate
            if hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
                z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
            else:
                z_spacing = getattr(slices[0], 'SliceThickness', 1.0)
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0)

        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(getattr(slices[0], 'ImagePositionPatient', (0.0, 0.0, 0.0)))

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
        print(f"[Loader] Fast scanning (Step={self.step}): {folder_path} ...")
        if callback: callback(0, f"Fast scanning (step={self.step})...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)
        if not files:
            raise FileNotFoundError("No valid DICOM/CT files found")

        print("[Loader] Sorting files by natural filename order...")
        if callback: callback(10, "Sorting files...")
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))

        selected_files = files[::self.step]
        print(f"[Loader] Selected {len(selected_files)} / {len(files)} files.")

        if callback: callback(20, f"Reading {len(selected_files)} slices...")
        slices = self._parallel_read_files(selected_files, callback)
        
        if not slices:
            raise ValueError("No valid slices loaded.")
        
        # Sort slices by ImagePositionPatient if available (ensures correct Z-order)
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
            try:
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
                z_start = float(slices[0].ImagePositionPatient[2])
                z_end = float(slices[-1].ImagePositionPatient[2])
                print(f"[Loader] Sorted by Z-position: {z_start:.2f} -> {z_end:.2f}")
            except Exception as e:
                print(f"[Loader] Warning: Could not sort by header position: {e}")

        if callback: callback(50, "Building downsampled volume...")
        volume, spacing, origin = self._build_volume_downsampled(slices, callback)

        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Type": "Fast/Downsampled",
            "SortMethod": "Header (ImagePositionPatient)" if hasattr(slices[0], 'ImagePositionPatient') else "Filename"
        }

        print(f"[Loader] Fast Load complete: {volume.shape}, Spacing: {spacing}")
        if callback: callback(100, "Fast load complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _build_volume_downsampled(self, slices: List[pydicom.dataset.FileDataset], 
                                   callback: Optional[Callable] = None) -> Tuple[np.ndarray, tuple, tuple]:
        pixel_spacing = slices[0].PixelSpacing

        if len(slices) > 1:
            if hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
                z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
            else:
                z_spacing = getattr(slices[0], 'SliceThickness', 1.0) * self.step
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0) * self.step

        spacing = (
            float(pixel_spacing[0]) * self.step,
            float(pixel_spacing[1]) * self.step,
            float(z_spacing)
        )
        origin = tuple(getattr(slices[0], 'ImagePositionPatient', (0.0, 0.0, 0.0)))

        base_shape = slices[0].pixel_array.shape
        new_h = base_shape[0] // self.step
        new_w = base_shape[1] // self.step

        img_shape = (len(slices), new_h, new_w)
        volume = np.empty(img_shape, dtype=np.float32)

        total = len(slices)
        for i, s in enumerate(slices):
            if callback and i % 10 == 0:
                percent = 50 + int(50 * i / total)
                callback(percent, f"Downsampling slice {i+1}/{total}...")
                
            slope = float(getattr(s, 'RescaleSlope', 1))
            intercept = float(getattr(s, 'RescaleIntercept', 0))
            arr = s.pixel_array[::self.step, ::self.step].astype(np.float32)
            arr = arr[:new_h, :new_w]
            if slope != 1.0:
                arr *= slope
            if intercept != 0.0:
                arr += intercept
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
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Sorting...")
        
        # Sort by filename first
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        # Optionally verify with headers
        if self.use_header_sort:
            files = self._verify_sort_with_headers(files, callback)
        
        print(f"[MemoryMappedLoader] Found {len(files)} valid slices")
        
        # Read first file for metadata
        first_full = pydicom.dcmread(files[0])
        rows, cols = first_full.pixel_array.shape
        num_slices = len(files)
        
        pixel_spacing = first_full.PixelSpacing
        if num_slices > 1:
            second_ds = pydicom.dcmread(files[1], stop_before_pixels=True)
            if hasattr(second_ds, 'ImagePositionPatient') and hasattr(first_full, 'ImagePositionPatient'):
                z_spacing = abs(float(second_ds.ImagePositionPatient[2]) - float(first_full.ImagePositionPatient[2]))
            else:
                z_spacing = getattr(first_full, 'SliceThickness', 1.0)
        else:
            z_spacing = getattr(first_full, 'SliceThickness', 1.0)
        
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(getattr(first_full, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
        
        mmap_file = os.path.join(self.cache_dir, f"ct_volume_{id(self)}_{num_slices}.dat")
        print(f"[MemoryMappedLoader] Creating memory-mapped file: {mmap_file}")
        if callback: callback(30, "Creating memory map...")
        
        shape = (num_slices, rows, cols)
        volume = np.memmap(mmap_file, dtype=np.float32, mode='w+', shape=shape)
        
        print(f"[MemoryMappedLoader] Loading {num_slices} slices into memory-mapped array...")
        for i, f in enumerate(files):
            if i % 10 == 0:
                if callback:
                    percent = 30 + int(70 * i / num_slices)
                    callback(percent, f"Mapping slice {i+1}/{num_slices}...")
            
            ds = pydicom.dcmread(f)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            arr = ds.pixel_array.astype(np.float32)
            if slope != 1.0:
                arr *= slope
            if intercept != 0.0:
                arr += intercept
            volume[i] = arr
        
        volume.flush()
        
        print(f"[MemoryMappedLoader] Complete: {volume.shape}, Memory-mapped to disk")
        
        metadata = {
            "SampleID": getattr(first_full, "PatientID", "Unknown"),
            "ScanType": getattr(first_full, "Modality", "CT"),
            "SliceCount": num_slices,
            "Type": "Memory-Mapped",
            "MmapFile": mmap_file
        }
        
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
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Sorting...")
        
        # Sort by filename
        files.sort(key=lambda f: _natural_sort_key(os.path.basename(f)))
        
        # Optionally verify with headers
        if self.use_header_sort:
            files = self._verify_sort_with_headers(files, callback)
        
        self._file_list = files
        
        if callback: callback(50, "Reading first slice metadata...")
        first = pydicom.dcmread(self._file_list[0])
        rows, cols = first.pixel_array.shape
        num_slices = len(self._file_list)
        
        pixel_spacing = first.PixelSpacing
        if num_slices > 1:
            second = pydicom.dcmread(self._file_list[1], stop_before_pixels=True)
            if hasattr(second, 'ImagePositionPatient') and hasattr(first, 'ImagePositionPatient'):
                z_spacing = abs(float(second.ImagePositionPatient[2]) - float(first.ImagePositionPatient[2]))
            else:
                z_spacing = getattr(first, 'SliceThickness', 1.0)
        else:
            z_spacing = getattr(first, 'SliceThickness', 1.0)
        
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(getattr(first, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
        
        self._metadata = {
            "SampleID": getattr(first, "PatientID", "Unknown"),
            "ScanType": getattr(first, "Modality", "CT"),
            "SliceCount": num_slices,
            "Type": "Chunked",
            "ChunkSize": self.chunk_size,
            "Dimensions": (num_slices, rows, cols)
        }
        
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
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            arr = ds.pixel_array.astype(np.float32)
            if slope != 1.0:
                arr *= slope
            if intercept != 0.0:
                arr += intercept
            chunk[i] = arr
        
        self._current_chunk = chunk
        self._current_chunk_range = (start_slice, end_slice)
        
        return chunk
    
    def get_total_slices(self) -> int:
        return len(self._file_list) if self._file_list else 0
    
    def get_num_chunks(self) -> int:
        total = self.get_total_slices()
        return (total + self.chunk_size - 1) // self.chunk_size
