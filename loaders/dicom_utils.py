"""DICOM loading utilities with GPU/Numba acceleration."""

import os
import re
import numpy as np
import pydicom
from glob import glob
from typing import List, Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def rescale_volume_numba(volume: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
        """Numba JIT accelerated volume rescaling."""
        for i in prange(volume.shape[0]):
            if slopes[i] != 1.0 or intercepts[i] != 0.0:
                for j in range(volume.shape[1]):
                    for k in range(volume.shape[2]):
                        volume[i, j, k] = volume[i, j, k] * slopes[i] + intercepts[i]
else:
    def rescale_volume_numba(volume, slopes, intercepts):
        for i in range(len(slopes)):
            if slopes[i] != 1.0:
                volume[i] *= slopes[i]
            if intercepts[i] != 0.0:
                volume[i] += intercepts[i]


def rescale_volume_gpu(volume_gpu, slopes: np.ndarray, intercepts: np.ndarray):
    """GPU-accelerated volume rescaling (in-place)."""
    import cupy as cp
    
    slopes_gpu = cp.asarray(slopes, dtype=cp.float32)
    intercepts_gpu = cp.asarray(intercepts, dtype=cp.float32)
    
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void rescale_kernel(float* volume, const float* slopes, const float* intercepts,
                        const int n_slices, const int slice_size) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n_slices * slice_size) return;
        int z = idx / slice_size;
        float s = slopes[z], b = intercepts[z];
        if (s != 1.0f || b != 0.0f) volume[idx] = volume[idx] * s + b;
    }
    ''', 'rescale_kernel')
    
    n_slices = volume_gpu.shape[0]
    slice_size = volume_gpu.shape[1] * volume_gpu.shape[2]
    blocks = (volume_gpu.size + 255) // 256
    kernel((blocks,), (256,), (volume_gpu, slopes_gpu, intercepts_gpu, n_slices, slice_size))
    del slopes_gpu, intercepts_gpu


def natural_sort_key(text: str):
    """Natural sort key for filenames (img_1, img_2, ..., img_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def validate_path(folder_path: str) -> None:
    """Validate folder path exists."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Path does not exist: {folder_path}")


def find_dicom_files(folder_path: str) -> List[str]:
    """Find DICOM files in folder (by extension then content check)."""
    files = glob(os.path.join(folder_path, "*.dcm"))
    if not files:
        files = [f for f in glob(os.path.join(folder_path, "*")) 
                 if os.path.isfile(f) and _is_dicom(f)]
    if not files:
        raise FileNotFoundError("No valid DICOM/CT files found")
    return files


def _is_dicom(filepath: str) -> bool:
    """Check if file is valid DICOM."""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False


def calculate_z_spacing(ds1, ds2=None, step: int = 1) -> float:
    """Calculate Z-spacing from DICOM headers."""
    if ds2 is not None:
        if hasattr(ds1, 'ImagePositionPatient') and hasattr(ds2, 'ImagePositionPatient'):
            return abs(float(ds2.ImagePositionPatient[2]) - float(ds1.ImagePositionPatient[2]))
    return float(getattr(ds1, 'SliceThickness', 1.0)) * step


def get_spacing_and_origin(ds1, ds2=None, xy_step: int = 1, z_step: int = 1) -> Tuple[tuple, tuple]:
    """Extract spacing and origin from DICOM dataset."""
    pixel_spacing = ds1.PixelSpacing
    z_spacing = calculate_z_spacing(ds1, ds2, z_step)
    spacing = (
        float(pixel_spacing[0]) * xy_step,
        float(pixel_spacing[1]) * xy_step,
        z_spacing
    )
    origin = tuple(getattr(ds1, 'ImagePositionPatient', (0.0, 0.0, 0.0)))
    return spacing, origin


def apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
    """Apply DICOM rescale slope/intercept to array (in-place)."""
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    if slope != 1.0:
        arr *= slope
    if intercept != 0.0:
        arr += intercept
    return arr


def sort_slices_by_position(slices: List, loader_name: str = "Loader") -> List:
    """Sort DICOM slices by Z-position."""
    if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            z_start = float(slices[0].ImagePositionPatient[2])
            z_end = float(slices[-1].ImagePositionPatient[2])
            print(f"[{loader_name}] Sorted by Z-position: {z_start:.2f} -> {z_end:.2f}")
        except Exception as e:
            print(f"[{loader_name}] Warning: Could not sort by header position: {e}")
    return slices


def extract_metadata(ds, slice_count: int, extra: dict = None) -> dict:
    """Extract common metadata from DICOM dataset."""
    metadata = {
        "SampleID": getattr(ds, "PatientID", "Unknown Sample"),
        "ScanType": getattr(ds, "Modality", "CT"),
        "SliceCount": slice_count,
        "SortMethod": "Header" if hasattr(ds, 'ImagePositionPatient') else "Filename"
    }
    if extra:
        metadata.update(extra)
    return metadata
