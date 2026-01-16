"""
Shared utility functions for DICOM loaders.
Provides common operations for loading and processing DICOM files.
"""

import os
import re
import numpy as np
import pydicom
from glob import glob
from typing import List, Tuple, Optional

# Try to import Numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ==========================================
# Numba-accelerated rescale
# ==========================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def rescale_volume_numba(volume: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
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
    def rescale_volume_numba(volume, slopes, intercepts):
        """Fallback: vectorized numpy rescale."""
        for i in range(len(slopes)):
            if slopes[i] != 1.0:
                volume[i] *= slopes[i]
            if intercepts[i] != 0.0:
                volume[i] += intercepts[i]


def natural_sort_key(text: str):
    """
    Natural sorting key for filenames like img_1, img_2, ..., img_10.
    
    Example:
        sorted(files, key=natural_sort_key)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


# ==========================================
# Path & File Validation
# ==========================================

def validate_path(folder_path: str) -> None:
    """
    Validate that folder path exists.
    
    Raises:
        FileNotFoundError: If path does not exist
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Path does not exist: {folder_path}")


def find_dicom_files(folder_path: str) -> List[str]:
    """
    Find DICOM files in folder, checking extension first then content.
    
    Args:
        folder_path: Directory to search
        
    Returns:
        List of file paths to valid DICOM files
        
    Raises:
        FileNotFoundError: If no valid DICOM files found
    """
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
                except (pydicom.errors.InvalidDicomError, IOError, OSError):
                    # Not a valid DICOM file, skip
                    continue
    if not files:
        raise FileNotFoundError("No valid DICOM/CT files found")
    return files


# ==========================================
# DICOM Header Utilities
# ==========================================

def calculate_z_spacing(ds1, ds2=None, step: int = 1) -> float:
    """
    Calculate Z-spacing from DICOM headers.
    
    Args:
        ds1: First DICOM dataset
        ds2: Second DICOM dataset (optional, for calculating from position difference)
        step: Multiplier for spacing (used by downsampling loaders)
        
    Returns:
        Z-spacing in mm
    """
    if ds2 is not None:
        if hasattr(ds1, 'ImagePositionPatient') and hasattr(ds2, 'ImagePositionPatient'):
            return abs(float(ds2.ImagePositionPatient[2]) - float(ds1.ImagePositionPatient[2]))
    return float(getattr(ds1, 'SliceThickness', 1.0)) * step


def get_spacing_and_origin(ds1, ds2=None, xy_step: int = 1, z_step: int = 1) -> Tuple[tuple, tuple]:
    """
    Extract spacing and origin from DICOM dataset.
    
    Args:
        ds1: First DICOM dataset
        ds2: Second DICOM dataset (for Z-spacing calculation)
        xy_step: Multiplier for XY spacing (used by downsampling)
        z_step: Multiplier for Z spacing (used by downsampling)
        
    Returns:
        Tuple of (spacing, origin) where each is (x, y, z)
    """
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
    """
    Apply DICOM rescale slope and intercept to pixel array (in-place).
    
    Args:
        arr: Pixel array (float32)
        ds: DICOM dataset with RescaleSlope/RescaleIntercept
        
    Returns:
        Modified array (same reference as input)
    """
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    if slope != 1.0:
        arr *= slope
    if intercept != 0.0:
        arr += intercept
    return arr


def sort_slices_by_position(slices: List, loader_name: str = "Loader") -> List:
    """
    Sort DICOM slices by ImagePositionPatient Z-coordinate.
    
    Args:
        slices: List of DICOM datasets
        loader_name: Name for logging
        
    Returns:
        Sorted list of slices
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


def extract_metadata(ds, slice_count: int, extra: dict = None) -> dict:
    """
    Extract common metadata from DICOM dataset.
    
    Args:
        ds: DICOM dataset
        slice_count: Number of slices
        extra: Additional metadata to include
        
    Returns:
        Metadata dictionary
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
