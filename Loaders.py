import os
import numpy as np
import pydicom
from glob import glob
from typing import List, Tuple, Optional, Callable
from scipy.ndimage import gaussian_filter
from core import BaseLoader, VolumeData


# ==========================================
# Data Loader Implementations
# ==========================================

class DicomSeriesLoader(BaseLoader):
    """Concrete DICOM series loader for Industrial CT/Micro-CT scans"""

    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Scanning sample folder: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Reading headers...")
        
        slices = self._read_and_sort_slices(files)
        if callback: callback(30, "Headers read. Building 3D volume...")

        volume, spacing, origin = self._build_volume(slices, callback)

        # Extract basic sample metadata
        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Description": getattr(slices[0], "StudyDescription", "No Description")
        }

        print(f"[Loader] Loading complete: {volume.shape}, Voxel Spacing: {spacing}")
        if callback: callback(100, "Loading complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _find_dicom_files(self, folder_path: str) -> List[str]:
        files = glob(os.path.join(folder_path, "*.dcm"))
        if not files:
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

    def _read_and_sort_slices(self, files: List[str]) -> List[pydicom.dataset.FileDataset]:
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                if hasattr(ds, 'ImagePositionPatient'):
                    slices.append(ds)
            except Exception as e:
                print(f"Warning: Skipping file {f} - {e}")

        # Sort by Z position to reconstruct the 3D volume correctly
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices

    def _build_volume(self, slices: List[pydicom.dataset.FileDataset], callback: Optional[Callable] = None) -> Tuple[np.ndarray, tuple, tuple]:
        pixel_spacing = slices[0].PixelSpacing
        if len(slices) > 1:
            z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0)

        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(slices[0].ImagePositionPatient)

        img_shape = list(slices[0].pixel_array.shape)
        img_shape.insert(0, len(slices))
        volume = np.zeros(img_shape, dtype=np.float32)

        total = len(slices)
        for i, s in enumerate(slices):
            if callback and i % 10 == 0:
                percent = 30 + int(70 * i / total)
                callback(percent, f"Loading slice {i+1}/{total}...")
                
            slope = getattr(s, 'RescaleSlope', 1)
            intercept = getattr(s, 'RescaleIntercept', 0)
            volume[i, :, :] = s.pixel_array * slope + intercept

        return volume, spacing, origin


class FastDicomLoader(DicomSeriesLoader):
    """
    Fast loader that downsamples data (lowers resolution).
    Essential for previewing large Micro-CT datasets.
    """

    def __init__(self, step: int = 2):
        """
        :param step: Downsample factor. 2 means half resolution (1/8th volume size).
        """
        self.step = step

    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Fast scanning (Step={self.step}): {folder_path} ...")
        if callback: callback(0, f"Fast scannning (step={self.step})...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)

        if not files:
            raise FileNotFoundError("No valid DICOM/CT files found")

        # Optimization: Try sorting by filename first (Natural Sort)
        import re

        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        print("[Loader] attempting natural filename sort...")
        if callback: callback(10, "Sorting files...")
        files.sort(key=lambda f: natural_keys(os.path.basename(f)))

        # Downsample: Select every nth file based on filename order
        selected_files = files[::self.step]
        print(f"[Loader] Selected {len(selected_files)} / {len(files)} files via filename sort.")

        # Read only selected slices
        slices = []
        total_selected = len(selected_files)
        for i, f in enumerate(selected_files):
            if callback and i % 20 == 0:
                percent = 10 + int(20 * i / total_selected)
                callback(percent, f"Reading headers {i}/{total_selected}...")
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception as e:
                print(f"Warning: Failed to read {f} - {e}")
        
        # Verify Z-Order consistency of the loaded subset
        if len(slices) > 1:
            z_start = float(slices[0].ImagePositionPatient[2])
            z_end = float(slices[-1].ImagePositionPatient[2])
            print(f"[Loader] Z-Range: {z_start} -> {z_end}")

        if not slices:
            raise ValueError("No valid slices loaded.")

        if callback: callback(30, "Headers read. Building downsampled volume...")
        volume, spacing, origin = self._build_volume_downsampled(slices, callback)

        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Type": "Fast/Downsampled (Filename Sorted)"
        }

        print(f"[Loader] Fast Load complete: {volume.shape}, Spacing: {spacing}")
        if callback: callback(100, "Fast load complete.")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _build_volume_downsampled(self, slices: List[pydicom.dataset.FileDataset], callback: Optional[Callable] = None) -> Tuple[np.ndarray, tuple, tuple]:
        # Calculate new spacing
        pixel_spacing = slices[0].PixelSpacing

        # Z-spacing check
        if len(slices) > 1:
            z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0) * self.step

        # New spacing increases by step factor
        spacing = (
            float(pixel_spacing[0]) * self.step,
            float(pixel_spacing[1]) * self.step,
            float(z_spacing)
        )
        origin = tuple(slices[0].ImagePositionPatient)

        # Determine new image shape (XY downsampling)
        base_shape = slices[0].pixel_array.shape
        new_h = base_shape[0] // self.step
        new_w = base_shape[1] // self.step

        img_shape = (len(slices), new_h, new_w)
        volume = np.zeros(img_shape, dtype=np.float32)

        total = len(slices)
        for i, s in enumerate(slices):
            if callback and i % 10 == 0:
                percent = 30 + int(70 * i / total)
                callback(percent, f"Downsampling slice {i+1}/{total}...")
                
            slope = getattr(s, 'RescaleSlope', 1)
            intercept = getattr(s, 'RescaleIntercept', 0)

            # Slicing [::step, ::step] performs the downsampling
            arr = s.pixel_array[::self.step, ::self.step]

            # Crop in case of slight shape mismatch due to integer division
            arr = arr[:new_h, :new_w]

            volume[i, :, :] = arr * slope + intercept

        return volume, spacing, origin



class DummyLoader(BaseLoader):
    """Synthetic porous media generator for testing"""

    def load(self, size: int = 128, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Generating synthetic internal porous structure (Size: {size})...")
        if callback: callback(0, "Generating random noise...")
        
        # 1. Generate random noise (Gaussian Random Field)
        np.random.seed(None)
        noise = np.random.rand(size, size, size)

        # 2. Apply Gaussian Blur to create organic, connected "blobs"
        sigma = 4.0
        print(f"[Loader] Applying Gaussian filter (sigma={sigma})...")
        if callback: callback(30, "Applying Gaussian filter...")
        blob_field = gaussian_filter(noise, sigma=sigma)

        # 3. Threshold to define Solid vs Void
        # Higher threshold = more void space
        threshold = np.mean(blob_field)

        volume = np.zeros((size, size, size), dtype=np.float32)

        # Solid matrix (High Intensity)
        if callback: callback(60, "Thresholding...")
        mask_solid = blob_field > threshold
        volume[mask_solid] = 1000

        # Void/Pore space (Low Intensity)
        mask_void = blob_field <= threshold
        volume[mask_void] = -1000

        # 4. Enforce Solid Boundary (Shell)
        # This ensures the pores are "Internal" and not touching the image border everywhere
        print("[Loader] Enforcing solid boundary shell...")
        if callback: callback(80, "Adding boundary shell...")
        border = 5
        volume[:border, :, :] = 1000
        volume[-border:, :, :] = 1000
        volume[:, :border, :] = 1000
        volume[:, -border:, :] = 1000
        volume[:, :, :border] = 1000
        volume[:, :, -border:] = 1000

        # 5. Add realistic scanner noise
        volume += np.random.normal(0, 50, (size, size, size))
        
        if callback: callback(100, "Generation complete.")
        return VolumeData(
            raw_data=volume,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            metadata={
                "Type": "Synthetic",
                "Description": "Solid Block with Internal Random Pores",
                "GenerationMethod": "Gaussian Random Field + Solid Shell"
            }
        )


# ==========================================
# Memory-Mapped Loader for Large Datasets
# ==========================================

class MemoryMappedDicomLoader(DicomSeriesLoader):
    """
    Memory-mapped loader for very large DICOM datasets.
    Uses numpy memory-mapping to avoid loading entire volume into RAM.
    Ideal for datasets > 4GB.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        :param cache_dir: Directory to store memory-mapped files. Uses temp dir if None.
        """
        import tempfile
        self.cache_dir = cache_dir or tempfile.gettempdir()
    
    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[MemoryMappedLoader] Scanning: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Reading metadata...")
        
        # Read first slice for metadata without loading pixels
        first_ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        
        # Sort files by position (read headers only)
        file_positions = []
        total_files = len(files)
        for i, f in enumerate(files):
            if callback and i % 50 == 0:
                percent = 10 + int(20 * i / total_files)
                callback(percent, f"Reading headers {i}/{total_files}...")
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    file_positions.append((f, float(ds.ImagePositionPatient[2])))
            except Exception as e:
                print(f"Warning: Skipping {f} - {e}")
        
        file_positions.sort(key=lambda x: x[1])
        sorted_files = [fp[0] for fp in file_positions]
        
        print(f"[MemoryMappedLoader] Found {len(sorted_files)} valid slices")
        
        # Get dimensions
        first_full = pydicom.dcmread(sorted_files[0])
        rows, cols = first_full.pixel_array.shape
        num_slices = len(sorted_files)
        
        # Calculate spacing
        pixel_spacing = first_full.PixelSpacing
        if num_slices > 1:
            second_ds = pydicom.dcmread(sorted_files[1], stop_before_pixels=True)
            z_spacing = abs(float(second_ds.ImagePositionPatient[2]) - float(first_full.ImagePositionPatient[2]))
        else:
            z_spacing = getattr(first_full, 'SliceThickness', 1.0)
        
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(first_full.ImagePositionPatient)
        
        # Create memory-mapped array
        import tempfile
        mmap_file = os.path.join(self.cache_dir, f"ct_volume_{id(self)}_{num_slices}.dat")
        print(f"[MemoryMappedLoader] Creating memory-mapped file: {mmap_file}")
        if callback: callback(30, "Creating memory map...")
        
        # Create the memory-mapped array
        shape = (num_slices, rows, cols)
        volume = np.memmap(mmap_file, dtype=np.float32, mode='w+', shape=shape)
        
        # Load slices into memory-mapped array
        print(f"[MemoryMappedLoader] Loading {num_slices} slices into memory-mapped array...")
        for i, f in enumerate(sorted_files):
            if i % 10 == 0:
                if callback:
                    percent = 30 + int(70 * i / num_slices)
                    callback(percent, f"Mapping slice {i+1}/{num_slices}...")
            
            ds = pydicom.dcmread(f)
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            volume[i, :, :] = ds.pixel_array * slope + intercept
        
        # Flush to disk
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
    Useful when you only need to process specific regions (ROI).
    """
    
    def __init__(self, chunk_size: int = 64):
        """
        :param chunk_size: Number of slices to load per chunk.
        """
        self.chunk_size = chunk_size
        self._file_list = None
        self._metadata = None
        self._current_chunk = None
        self._current_chunk_range = None
    
    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        """Load metadata and prepare for chunked access."""
        print(f"[ChunkedLoader] Preparing chunked access: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Reading headers...")
        
        # Sort files by Z position
        file_positions = []
        total_files = len(files)
        for i, f in enumerate(files):
            if callback and i % 50 == 0:
                percent = 10 + int(30 * i / total_files)
                callback(percent, f"Reading header {i}/{total_files}...")
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    file_positions.append((f, float(ds.ImagePositionPatient[2])))
            except Exception:
                continue
        
        file_positions.sort(key=lambda x: x[1])
        self._file_list = [fp[0] for fp in file_positions]
        
        if callback: callback(50, "Headers read. Reading first slice metadata...")
        # Read first slice for dimensions
        first = pydicom.dcmread(self._file_list[0])
        rows, cols = first.pixel_array.shape
        num_slices = len(self._file_list)
        
        # Calculate spacing
        pixel_spacing = first.PixelSpacing
        if num_slices > 1:
            second = pydicom.dcmread(self._file_list[1], stop_before_pixels=True)
            z_spacing = abs(float(second.ImagePositionPatient[2]) - float(first.ImagePositionPatient[2]))
        else:
            z_spacing = getattr(first, 'SliceThickness', 1.0)
        
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(first.ImagePositionPatient)
        
        self._metadata = {
            "SampleID": getattr(first, "PatientID", "Unknown"),
            "ScanType": getattr(first, "Modality", "CT"),
            "SliceCount": num_slices,
            "Type": "Chunked",
            "ChunkSize": self.chunk_size,
            "Dimensions": (num_slices, rows, cols)
        }
        
        # Load first chunk as preview
        print(f"[ChunkedLoader] Loading first chunk (0-{min(self.chunk_size, num_slices)})...")
        if callback: callback(80, "Loading initial preview chunk...")
        chunk = self.load_chunk(0)
        
        print(f"[ChunkedLoader] Ready. Total slices: {num_slices}, Chunk size: {self.chunk_size}")
        
        if callback: callback(100, "Initialization complete.")
        return VolumeData(raw_data=chunk, spacing=spacing, origin=origin, metadata=self._metadata)
    
    def load_chunk(self, start_slice: int) -> np.ndarray:
        """Load a specific chunk of slices."""
        if self._file_list is None:
            raise RuntimeError("Must call load() first to initialize the loader")
        
        end_slice = min(start_slice + self.chunk_size, len(self._file_list))
        
        # Check if chunk is already loaded
        if self._current_chunk_range == (start_slice, end_slice):
            return self._current_chunk
        
        print(f"[ChunkedLoader] Loading chunk: slices {start_slice}-{end_slice}")
        
        first = pydicom.dcmread(self._file_list[0])
        rows, cols = first.pixel_array.shape
        chunk_size = end_slice - start_slice
        
        chunk = np.zeros((chunk_size, rows, cols), dtype=np.float32)
        
        for i, idx in enumerate(range(start_slice, end_slice)):
            ds = pydicom.dcmread(self._file_list[idx])
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            chunk[i, :, :] = ds.pixel_array * slope + intercept
        
        self._current_chunk = chunk
        self._current_chunk_range = (start_slice, end_slice)
        
        return chunk
    
    def get_total_slices(self) -> int:
        """Return total number of slices available."""
        return len(self._file_list) if self._file_list else 0
    
    def get_num_chunks(self) -> int:
        """Return total number of chunks."""
        total = self.get_total_slices()
        return (total + self.chunk_size - 1) // self.chunk_size