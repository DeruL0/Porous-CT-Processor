"""
Pore extraction processor for segmenting void space from solid matrix.
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
from typing import Optional, Callable

from core import BaseProcessor, VolumeData


class PoreExtractionProcessor(BaseProcessor):
    """
    Segments the void space (pores) from the solid matrix.
    Returns a Voxel-based VolumeData object representing "Air".
    """

    @staticmethod
    def suggest_threshold(data: VolumeData) -> int:
        """Calculate optimal threshold using Otsu's method."""
        if data.raw_data is None:
            return -300
        
        clean_data = data.raw_data[np.isfinite(data.raw_data)]
        if clean_data.size == 0:
            return -300
            
        try:
            thresh = threshold_otsu(clean_data)
            return int(thresh)
        except Exception as e:
            print(f"Otsu failed: {e}")
            return -300

    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None,
                threshold: int = -300) -> VolumeData:
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        def report(p: int, msg: str):
            print(f"[Processor] {msg}")
            if callback: callback(p, msg)

        report(0, f"Starting pore detection (Threshold < {threshold})...")

        # 1. Binarization (Air vs Solid)
        solid_mask = data.raw_data > threshold
        report(20, "Binarization complete. Filling holes...")

        # 2. Morphology
        filled_volume = ndimage.binary_fill_holes(solid_mask)
        pores_mask = filled_volume ^ solid_mask
        report(50, "Morphology operations complete. Calculating stats...")

        # Quantitative Analysis
        pore_voxels = np.sum(pores_mask)
        total_voxels = data.raw_data.size
        porosity_pct = (pore_voxels / total_voxels) * 100.0

        # Label connected components
        report(70, "Labeling connected components (this may take a while)...")
        labeled_array, num_features = ndimage.label(pores_mask, structure=np.ones((3, 3, 3)))

        report(90, f"Found {num_features} pores. Generating output volume...")

        # Create Output Volume (Voxel Grid)
        processed_volume = np.zeros_like(data.raw_data)
        processed_volume[pores_mask] = 1000  # Highlight Pores

        report(100, "Processing complete.")

        return VolumeData(
            raw_data=processed_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Void Volume",
                "Porosity": f"{porosity_pct:.2f}%",
                "PoreCount": int(num_features)
            }
        )
