"""
Pore extraction processor for segmenting void space from solid matrix.
Memory-optimized for large volumes using chunked processing.
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
from typing import Optional, Callable
import gc

from core import BaseProcessor, VolumeData
from config import (
    PROCESS_CHUNK_THRESHOLD,
    PROCESS_CHUNK_SIZE,
    PROCESS_OTSU_SAMPLE_THRESHOLD
)

# Alias for backward compatibility
CHUNK_THRESHOLD = PROCESS_CHUNK_THRESHOLD


class PoreExtractionProcessor(BaseProcessor):
    """
    Segments the void space (pores) from the solid matrix.
    Returns a Voxel-based VolumeData object representing "Air".
    
    Memory-optimized: uses chunked processing for large volumes.
    """

    def __init__(self, chunk_size: int = PROCESS_CHUNK_SIZE):
        """
        Args:
            chunk_size: Number of slices to process at a time for large volumes.
        """
        self.chunk_size = chunk_size

    @staticmethod
    def suggest_threshold(data: VolumeData) -> int:
        """Calculate optimal threshold using Otsu's method."""
        if data.raw_data is None:
            return -300
        
        # Sample data for large volumes to save memory
        raw = data.raw_data
        if raw.size > PROCESS_OTSU_SAMPLE_THRESHOLD:
            # Sample every 4th voxel
            sample = raw[::4, ::4, ::4].flatten()
        else:
            sample = raw.flatten()
        
        clean_data = sample[np.isfinite(sample)]
        if clean_data.size == 0:
            return -300
            
        try:
            thresh = threshold_otsu(clean_data)
            del sample, clean_data
            gc.collect()
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

        raw = data.raw_data
        volume_bytes = raw.nbytes
        
        # Check if we need chunked processing
        if volume_bytes > CHUNK_THRESHOLD:
            return self._process_chunked(data, threshold, report)
        else:
            return self._process_standard(data, threshold, report)

    def _process_standard(self, data: VolumeData, threshold: int, report) -> VolumeData:
        """Standard processing for smaller volumes."""
        raw = data.raw_data
        
        # 1. Binarization (Air vs Solid)
        solid_mask = raw > threshold
        report(20, "Binarization complete. Filling holes...")

        # 2. Morphology
        filled_volume = ndimage.binary_fill_holes(solid_mask)
        pores_mask = filled_volume ^ solid_mask
        
        # Free intermediate arrays
        del solid_mask, filled_volume
        gc.collect()
        
        report(50, "Morphology operations complete. Calculating stats...")

        # Quantitative Analysis
        pore_voxels = np.sum(pores_mask)
        total_voxels = raw.size
        porosity_pct = (pore_voxels / total_voxels) * 100.0

        # Label connected components
        report(70, "Labeling connected components...")
        labeled_array, num_features = ndimage.label(pores_mask, structure=np.ones((3, 3, 3)))
        del labeled_array
        gc.collect()

        report(90, f"Found {num_features} pores. Generating output volume...")

        # Create Output Volume
        processed_volume = np.zeros_like(raw)
        processed_volume[pores_mask] = 1000
        del pores_mask
        gc.collect()

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

    def _process_chunked(self, data: VolumeData, threshold: int, report) -> VolumeData:
        """Chunked processing for large volumes to avoid OOM."""
        raw = data.raw_data
        n_slices = raw.shape[0]
        chunk_size = self.chunk_size
        n_chunks = (n_slices + chunk_size - 1) // chunk_size
        
        report(5, f"Large volume detected. Processing in {n_chunks} chunks...")
        
        # Pre-allocate output with same shape (will overwrite)
        processed_volume = np.zeros(raw.shape, dtype=np.float32)
        
        total_pore_voxels = 0
        total_voxels = raw.size
        
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_slices)
            
            # Process chunk
            chunk = raw[start:end]
            
            # Binarization
            solid_mask = chunk > threshold
            
            # Fill holes (2D per slice for memory efficiency)
            pores_mask = np.zeros_like(solid_mask)
            for j in range(solid_mask.shape[0]):
                filled = ndimage.binary_fill_holes(solid_mask[j])
                pores_mask[j] = filled ^ solid_mask[j]
            
            # Count pores
            total_pore_voxels += np.sum(pores_mask)
            
            # Write to output
            processed_volume[start:end][pores_mask] = 1000
            
            # Free memory
            del chunk, solid_mask, pores_mask
            gc.collect()
            
            progress = 10 + int(80 * (i + 1) / n_chunks)
            report(progress, f"Processed chunk {i+1}/{n_chunks}...")
        
        porosity_pct = (total_pore_voxels / total_voxels) * 100.0
        
        report(95, "Estimating pore count...")
        # Simplified pore counting for large volumes (sample middle chunk)
        mid_start = (n_slices // 2) - chunk_size // 2
        mid_end = mid_start + chunk_size
        mid_chunk = processed_volume[mid_start:mid_end] > 0
        _, estimated_features = ndimage.label(mid_chunk, structure=np.ones((3, 3, 3)))
        # Extrapolate
        estimated_total = int(estimated_features * n_chunks / 2)
        del mid_chunk
        gc.collect()

        report(100, "Chunked processing complete.")

        return VolumeData(
            raw_data=processed_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Void Volume (Chunked)",
                "Porosity": f"{porosity_pct:.2f}%",
                "PoreCount": f"~{estimated_total} (estimated)"
            }
        )

