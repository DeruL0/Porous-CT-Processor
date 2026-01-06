"""
DICOM series loaders for CT/Micro-CT data.
"""

import os
import re
import numpy as np
import pydicom
from glob import glob
from typing import List, Tuple, Optional, Callable

from core import BaseLoader, VolumeData


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
        self.step = step

    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Fast scanning (Step={self.step}): {folder_path} ...")
        if callback: callback(0, f"Fast scanning (step={self.step})...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)
        if not files:
            raise FileNotFoundError("No valid DICOM/CT files found")

        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        print("[Loader] Attempting natural filename sort...")
        if callback: callback(10, "Sorting files...")
        files.sort(key=lambda f: natural_keys(os.path.basename(f)))

        selected_files = files[::self.step]
        print(f"[Loader] Selected {len(selected_files)} / {len(files)} files via filename sort.")

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
        pixel_spacing = slices[0].PixelSpacing

        if len(slices) > 1:
            z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        else:
            z_spacing = getattr(slices[0], 'SliceThickness', 1.0) * self.step

        spacing = (
            float(pixel_spacing[0]) * self.step,
            float(pixel_spacing[1]) * self.step,
            float(z_spacing)
        )
        origin = tuple(slices[0].ImagePositionPatient)

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
            arr = s.pixel_array[::self.step, ::self.step]
            arr = arr[:new_h, :new_w]
            volume[i, :, :] = arr * slope + intercept

        return volume, spacing, origin


class MemoryMappedDicomLoader(DicomSeriesLoader):
    """
    Memory-mapped loader for very large DICOM datasets.
    Uses numpy memory-mapping to avoid loading entire volume into RAM.
    """
    
    def __init__(self, cache_dir: str = None):
        import tempfile
        self.cache_dir = cache_dir or tempfile.gettempdir()
    
    def load(self, folder_path: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[MemoryMappedLoader] Scanning: {folder_path} ...")
        if callback: callback(0, "Scanning directory...")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")
        
        files = self._find_dicom_files(folder_path)
        if callback: callback(10, f"Found {len(files)} files. Reading metadata...")
        
        first_ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        
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
        
        first_full = pydicom.dcmread(sorted_files[0])
        rows, cols = first_full.pixel_array.shape
        num_slices = len(sorted_files)
        
        pixel_spacing = first_full.PixelSpacing
        if num_slices > 1:
            second_ds = pydicom.dcmread(sorted_files[1], stop_before_pixels=True)
            z_spacing = abs(float(second_ds.ImagePositionPatient[2]) - float(first_full.ImagePositionPatient[2]))
        else:
            z_spacing = getattr(first_full, 'SliceThickness', 1.0)
        
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing))
        origin = tuple(first_full.ImagePositionPatient)
        
        mmap_file = os.path.join(self.cache_dir, f"ct_volume_{id(self)}_{num_slices}.dat")
        print(f"[MemoryMappedLoader] Creating memory-mapped file: {mmap_file}")
        if callback: callback(30, "Creating memory map...")
        
        shape = (num_slices, rows, cols)
        volume = np.memmap(mmap_file, dtype=np.float32, mode='w+', shape=shape)
        
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
    
    def __init__(self, chunk_size: int = 64):
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
        if callback: callback(10, f"Found {len(files)} files. Reading headers...")
        
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
        first = pydicom.dcmread(self._file_list[0])
        rows, cols = first.pixel_array.shape
        num_slices = len(self._file_list)
        
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
        return len(self._file_list) if self._file_list else 0
    
    def get_num_chunks(self) -> int:
        total = self.get_total_slices()
        return (total + self.chunk_size - 1) // self.chunk_size
