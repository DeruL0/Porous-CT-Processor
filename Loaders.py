import os
import numpy as np
import pydicom
from glob import glob
from typing import List, Tuple
from scipy.ndimage import gaussian_filter
from Core import BaseLoader, VolumeData


# ==========================================
# Data Loader Implementations
# ==========================================

class DicomSeriesLoader(BaseLoader):
    """Concrete DICOM series loader for Industrial CT/Micro-CT scans"""

    def load(self, folder_path: str) -> VolumeData:
        print(f"[Loader] Scanning sample folder: {folder_path} ...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)
        slices = self._read_and_sort_slices(files)

        volume, spacing, origin = self._build_volume(slices)

        # Extract basic sample metadata
        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Description": getattr(slices[0], "StudyDescription", "No Description")
        }

        print(f"[Loader] Loading complete: {volume.shape}, Voxel Spacing: {spacing}")
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

    def _build_volume(self, slices: List[pydicom.dataset.FileDataset]) -> Tuple[np.ndarray, tuple, tuple]:
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

        for i, s in enumerate(slices):
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

    def load(self, folder_path: str) -> VolumeData:
        print(f"[Loader] Fast scanning (Step={self.step}): {folder_path} ...")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path does not exist: {folder_path}")

        files = self._find_dicom_files(folder_path)

        # Optimization: Only read headers for sorting
        print("[Loader] Reading headers for sorting...")
        file_z_pairs = []
        for f in files:
            try:
                # stop_before_pixels=True significantly speeds up scanning
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    file_z_pairs.append((f, float(ds.ImagePositionPatient[2])))
            except:
                continue

        # Sort by Z position
        file_z_pairs.sort(key=lambda x: x[1])

        # Downsample Z-axis (Select every nth file)
        selected_files = [pair[0] for pair in file_z_pairs[::self.step]]
        print(f"[Loader] Selected {len(selected_files)} / {len(files)} slices for loading.")

        # Read full data only for selected slices
        slices = []
        for f in selected_files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception as e:
                print(f"Warning: Failed to read {f} - {e}")

        if not slices:
            raise ValueError("No valid slices loaded.")

        volume, spacing, origin = self._build_volume_downsampled(slices)

        metadata = {
            "SampleID": getattr(slices[0], "PatientID", "Unknown Sample"),
            "ScanType": getattr(slices[0], "Modality", "CT"),
            "SliceCount": len(slices),
            "Type": "Fast/Downsampled"
        }

        print(f"[Loader] Fast Load complete: {volume.shape}, Spacing: {spacing}")
        return VolumeData(raw_data=volume, spacing=spacing, origin=origin, metadata=metadata)

    def _build_volume_downsampled(self, slices: List[pydicom.dataset.FileDataset]) -> Tuple[np.ndarray, tuple, tuple]:
        # Calculate new spacing
        pixel_spacing = slices[0].PixelSpacing

        # Z-spacing check
        if len(slices) > 1:
            # Calculate actual distance between the sampled slices
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

        for i, s in enumerate(slices):
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

    def load(self, size: int = 128) -> VolumeData:
        print(f"[Loader] Generating synthetic internal porous structure (Size: {size})...")

        # 1. Generate random noise (Gaussian Random Field)
        np.random.seed(None)
        noise = np.random.rand(size, size, size)

        # 2. Apply Gaussian Blur to create organic, connected "blobs"
        sigma = 4.0
        print(f"[Loader] Applying Gaussian filter (sigma={sigma})...")
        blob_field = gaussian_filter(noise, sigma=sigma)

        # 3. Threshold to define Solid vs Void
        # Higher threshold = more void space
        threshold = np.mean(blob_field)

        volume = np.zeros((size, size, size), dtype=np.float32)

        # Solid matrix (High Intensity)
        mask_solid = blob_field > threshold
        volume[mask_solid] = 1000

        # Void/Pore space (Low Intensity)
        mask_void = blob_field <= threshold
        volume[mask_void] = -1000

        # 4. Enforce Solid Boundary (Shell)
        # This ensures the pores are "Internal" and not touching the image border everywhere
        print("[Loader] Enforcing solid boundary shell...")
        border = 5
        volume[:border, :, :] = 1000
        volume[-border:, :, :] = 1000
        volume[:, :border, :] = 1000
        volume[:, -border:, :] = 1000
        volume[:, :, :border] = 1000
        volume[:, :, -border:] = 1000

        # 5. Add realistic scanner noise
        volume += np.random.normal(0, 50, (size, size, size))

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