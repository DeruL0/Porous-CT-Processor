"""
Synthetic data generators for testing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Callable

from core import BaseLoader, VolumeData


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
        volume += np.random.normal(0, 50, (size, size, size)).astype(np.float32)
        
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
