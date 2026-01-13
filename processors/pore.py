"""
Pore extraction processor for segmenting void space from solid matrix.
Memory-optimized for large volumes using chunked processing.
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
from typing import Optional, Callable, Tuple
import gc

from core import BaseProcessor, VolumeData
from processors.utils import binary_fill_holes
from config import (
    PROCESS_CHUNK_THRESHOLD,
    PROCESS_CHUNK_SIZE,
    PROCESS_OTSU_SAMPLE_THRESHOLD
)

# Try to import cc3d for faster 3D labeling (10-50x speedup)
try:
    import cc3d
    HAS_CC3D = True
    print("[PoreProcessor] Using cc3d for fast 3D labeling")
except ImportError:
    HAS_CC3D = False
    print("[PoreProcessor] cc3d not found, using scipy.ndimage.label (slower)")

# Alias for backward compatibility
CHUNK_THRESHOLD = PROCESS_CHUNK_THRESHOLD


def fast_label(binary_mask: np.ndarray, connectivity: int = 26) -> Tuple[np.ndarray, int]:
    """
    Fast 3D connected component labeling.
    Uses cc3d if available (10-50x faster), otherwise falls back to scipy.
    
    Args:
        binary_mask: Binary 3D array (True = foreground)
        connectivity: 6, 18, or 26 for 3D (only used by cc3d)
        
    Returns:
        (labeled_array, num_features)
    """
    if HAS_CC3D:
        # cc3d is MUCH faster for large 3D arrays
        labeled = cc3d.connected_components(binary_mask, connectivity=connectivity)
        num_features = int(labeled.max())
        return labeled, num_features
    else:
        # Fallback to scipy (slower but always available)
        structure = np.ones((3, 3, 3)) if connectivity == 26 else None
        return ndimage.label(binary_mask, structure=structure)


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
    def suggest_threshold(data: VolumeData, algorithm: str = 'auto') -> int:
        """
        Calculate optimal threshold using specified algorithm.
        
        Args:
            data: Volume data to analyze
            algorithm: One of 'auto', 'otsu', 'li', 'yen', 'triangle', 'minimum'
                - auto: Smart selection based on histogram bimodality
                - otsu: Classic bimodal thresholding
                - li: Minimum cross-entropy (good for noisy data)
                - yen: Maximum correlation
                - triangle: Good for CT 'peak+tail' histograms
                - minimum: Valley between peaks
                
        Returns:
            Suggested threshold value (int)
        """
        from skimage.filters import (
            threshold_otsu, threshold_li, threshold_yen, threshold_triangle,
            threshold_minimum
        )
        
        if data.raw_data is None:
            return -300
        
        raw = data.raw_data
        
        # Sample data for large volumes
        if raw.size > PROCESS_OTSU_SAMPLE_THRESHOLD:
            sample = raw[::4, ::4, ::4].flatten()
        else:
            sample = raw.flatten()
        
        clean_data = sample[np.isfinite(sample)]
        if clean_data.size == 0:
            return -300
        
        try:
            # Compute histogram for analysis
            hist, bin_edges = np.histogram(clean_data, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Normalize histogram
            hist_norm = hist.astype(float) / hist.sum()
            
            # Calculate bimodality coefficient
            # Values > 0.555 indicate bimodal distribution
            n = len(clean_data)
            mean = np.mean(clean_data)
            std = np.std(clean_data)
            if std > 0:
                skewness = np.mean(((clean_data - mean) / std) ** 3)
                kurtosis = np.mean(((clean_data - mean) / std) ** 4) - 3
                bimodality = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            else:
                bimodality = 0
            
            # Calculate thresholds using multiple methods
            threshold_methods = {
                'otsu': threshold_otsu,
                'li': threshold_li,
                'yen': threshold_yen,
                'triangle': threshold_triangle,
                'minimum': threshold_minimum
            }
            
            valid_thresholds = {}
            for name, func in threshold_methods.items():
                try:
                    valid_thresholds[name] = func(clean_data)
                except Exception:
                    pass  # Skip failed methods
            
            if not valid_thresholds:
                return -300
            
            # If specific algorithm requested, use it directly
            if algorithm != 'auto' and algorithm in valid_thresholds:
                selected_thresh = valid_thresholds[algorithm]
                print(f"[Threshold] Using {algorithm} = {int(selected_thresh)}")
                del sample, clean_data
                gc.collect()
                return int(selected_thresh)
            elif algorithm != 'auto':
                # Requested algorithm failed, fall back to auto
                print(f"[Threshold] {algorithm} failed, falling back to auto")
            
            # AUTO MODE: Select best threshold based on histogram characteristics
            best_method = 'otsu'
            
            if bimodality > 0.6:
                # Strongly bimodal: prefer minimum or otsu
                if 'minimum' in valid_thresholds:
                    best_method = 'minimum'
                elif 'otsu' in valid_thresholds:
                    best_method = 'otsu'
            elif bimodality > 0.4:
                # Moderately bimodal: prefer Li (minimum cross-entropy)
                if 'li' in valid_thresholds:
                    best_method = 'li'
                elif 'otsu' in valid_thresholds:
                    best_method = 'otsu'
            else:
                # Weak bimodality or unimodal: prefer triangle for CT
                # Triangle works well for "peak + tail" distribution common in CT
                if 'triangle' in valid_thresholds:
                    best_method = 'triangle'
                elif 'yen' in valid_thresholds:
                    best_method = 'yen'
                elif 'li' in valid_thresholds:
                    best_method = 'li'
            
            # Validate threshold is reasonable
            data_min, data_max = clean_data.min(), clean_data.max()
            data_range = data_max - data_min
            
            selected_thresh = valid_thresholds[best_method]
            
            # Ensure threshold is not at extreme edges (likely failure)
            if selected_thresh < data_min + 0.05 * data_range:
                # Too low, try other methods
                for method in ['li', 'yen', 'otsu']:
                    if method in valid_thresholds:
                        alt = valid_thresholds[method]
                        if alt > data_min + 0.05 * data_range:
                            selected_thresh = alt
                            best_method = method
                            break
            
            if selected_thresh > data_max - 0.05 * data_range:
                # Too high, try other methods
                for method in ['triangle', 'li', 'otsu']:
                    if method in valid_thresholds:
                        alt = valid_thresholds[method]
                        if alt < data_max - 0.05 * data_range:
                            selected_thresh = alt
                            best_method = method
                            break
            
            print(f"[AutoThreshold] Bimodality: {bimodality:.3f}, Selected: {best_method} = {int(selected_thresh)}")
            print(f"[AutoThreshold] All: {', '.join(f'{k}={int(v)}' for k,v in valid_thresholds.items())}")
            
            del sample, clean_data
            gc.collect()
            
            return int(selected_thresh)
            
        except Exception as e:
            print(f"Advanced threshold failed: {e}, falling back to median")
            # Ultimate fallback: use median
            median_val = np.median(clean_data)
            del sample, clean_data
            gc.collect()
            return int(median_val)

    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None,
                threshold: int = -300) -> VolumeData:
        """
        Extract pores from volume data.
        
        Uses shared cache: if PNM already computed pores_mask, reuse it.
        Also stores result in cache for PNM to reuse.
        """
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        from data.disk_cache import get_segmentation_cache
        
        def report(p: int, msg: str):
            print(f"[Processor] {msg}")
            if callback: callback(p, msg)

        cache = get_segmentation_cache()
        volume_id = f"{id(data.raw_data)}_{threshold}"
        
        # Check if we already have cached segmentation
        cached_mask = cache.get_pores_mask(volume_id)
        if cached_mask is not None:
            report(0, "Cache hit! Reusing segmentation from previous run...")
            pores_mask = cached_mask > 0  # Convert memmap to bool if needed
            cached_meta = cache.get_metadata(volume_id) or {}
            
            report(70, "Labeling connected components (cc3d)..." if HAS_CC3D else "Labeling connected components...")
            labeled_array, num_features = fast_label(pores_mask, connectivity=26)
            del labeled_array
            gc.collect()
            
            report(90, f"Found {num_features} pores. Generating output...")
            processed_volume = np.zeros(data.raw_data.shape, dtype=np.float32)
            processed_volume[pores_mask] = 1000
            
            porosity = cached_meta.get('porosity', 0)
            
            return VolumeData(
                raw_data=processed_volume,
                spacing=data.spacing,
                origin=data.origin,
                metadata={
                    "Type": "Processed - Void Volume",
                    "Porosity": f"{porosity:.2f}%",
                    "PoreCount": int(num_features),
                    "CacheHit": True
                }
            )

        report(0, f"Starting pore detection (Threshold < {threshold})...")

        raw = data.raw_data
        volume_bytes = raw.nbytes
        
        # Check if we need chunked processing
        if volume_bytes > CHUNK_THRESHOLD:
            return self._process_chunked(data, threshold, report, cache, volume_id)
        else:
            return self._process_standard(data, threshold, report, cache, volume_id)

    def _process_standard(self, data: VolumeData, threshold: int, report, 
                          cache=None, volume_id=None) -> VolumeData:
        """Standard processing for smaller volumes."""
        raw = data.raw_data
        
        # 1. Binarization (Air vs Solid)
        solid_mask = raw > threshold
        report(20, "Binarization complete. Filling holes...")

        # 2. Morphology
        filled_volume = binary_fill_holes(solid_mask)
        pores_mask = filled_volume ^ solid_mask
        
        # Free intermediate arrays
        del solid_mask, filled_volume
        gc.collect()
        
        report(50, "Morphology operations complete. Calculating stats...")

        # Quantitative Analysis
        pore_voxels = np.sum(pores_mask)
        total_voxels = raw.size
        porosity_pct = (pore_voxels / total_voxels) * 100.0

        # Label connected components (using cc3d if available for 10-50x speedup)
        report(70, "Labeling connected components (cc3d)..." if HAS_CC3D else "Labeling connected components...")
        labeled_array, num_features = fast_label(pores_mask, connectivity=26)
        del labeled_array
        gc.collect()

        report(90, f"Found {num_features} pores. Generating output volume...")

        # Store in shared cache for PNM to reuse
        if cache is not None and volume_id is not None:
            cache.store_pores_mask(volume_id, pores_mask, metadata={'porosity': porosity_pct})

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

    def _process_chunked(self, data: VolumeData, threshold: int, report,
                          cache=None, volume_id=None) -> VolumeData:
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
                filled = binary_fill_holes(solid_mask[j])
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
        _, estimated_features = fast_label(mid_chunk, connectivity=26)
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

