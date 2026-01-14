"""
Scientific data manager for porous media analysis workflow.

Responsibilities:
- Manages runtime data state (raw, segmented, PNM, ROI)
- Provides data manipulation operations (clip, invert, histogram)
- Coordinates cache lifecycle
- Emits signals on data changes for UI synchronization
"""

import gc
from typing import Optional, Tuple, Callable
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from core import VolumeData


class ScientificDataManager(QObject):
    """
    Central Data Management Class for Scientific Computing.

    Manages the critical stages of the Porous Media Analysis workflow:
    1. Raw CT Data (Original Scan)
    2. Segmented Volume (Distinguishing Air/Void from Solid)
    3. Pore Network Model (PNM) Mesh (Distinguishing Pores vs Throats)
    4. ROI Data (Region of Interest extracted sub-volume)
    
    Also provides data manipulation operations with proper cache coordination.
    """
    
    # Signals
    data_changed = pyqtSignal()  # Emitted when active data is modified
    data_loaded = pyqtSignal(object)  # Emitted when new data is loaded (VolumeData)

    def __init__(self):
        super().__init__()
        self.raw_ct_data: Optional[VolumeData] = None
        self.segmented_volume: Optional[VolumeData] = None
        self.pnm_model: Optional[VolumeData] = None
        self.roi_data: Optional[VolumeData] = None

    @property
    def active_data(self) -> Optional[VolumeData]:
        """
        Returns the most relevant data for processing.
        Priority: ROI > Raw (we process on extracted region if available)
        """
        if self.roi_data is not None:
            return self.roi_data
        return self.raw_ct_data

    def load_raw_data(self, data: VolumeData):
        """Sets the raw input data. Clears previous data first."""
        # Clear all previous data to free memory
        self.raw_ct_data = None
        self.segmented_volume = None
        self.pnm_model = None
        self.roi_data = None
        
        # Clear segmentation cache
        self._clear_segmentation_cache()
        
        # Force garbage collection before loading new data
        gc.collect()
        
        # Now set new data
        self.raw_ct_data = data
        self.data_loaded.emit(data)

    def set_segmented_data(self, data: VolumeData):
        """Stores the intermediate segmented void space."""
        self.segmented_volume = data

    def set_pnm_data(self, data: VolumeData):
        """Stores the generated Pore Network Model mesh."""
        self.pnm_model = data

    def set_roi_data(self, data: VolumeData):
        """Stores ROI-extracted sub-volume for focused analysis."""
        self.roi_data = data
        self.segmented_volume = None
        self.pnm_model = None
        self._clear_segmentation_cache()

    def clear_roi(self):
        """Clears ROI data, reverting to full raw volume."""
        self.roi_data = None
        self.segmented_volume = None
        self.pnm_model = None
        self._clear_segmentation_cache()

    # ==========================================
    # Data Manipulation Methods
    # ==========================================
    
    def clip_data(self, min_val: float, max_val: float, 
                  chunk_size: int = 32,
                  progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """
        Permanently clip active data to specified range.
        
        Args:
            min_val: Minimum value to clip to
            max_val: Maximum value to clip to  
            chunk_size: Number of slices to process per chunk
            progress_callback: Optional callback for progress updates (percent, message)
        
        Raises:
            ValueError: If no active data available
        """
        data = self.active_data
        if data is None or data.raw_data is None:
            raise ValueError("No active data to clip")
        
        raw = data.raw_data
        n_slices = raw.shape[0]
        n_chunks = (n_slices + chunk_size - 1) // chunk_size
        
        if progress_callback:
            progress_callback(0, "Clipping data...")
        
        # Chunked processing to avoid memory spikes
        for i in range(0, n_slices, chunk_size):
            end = min(i + chunk_size, n_slices)
            raw[i:end] = raw[i:end].clip(min_val, max_val)
            gc.collect()
            
            if progress_callback:
                chunk_idx = i // chunk_size
                percent = int(80 * (chunk_idx + 1) / n_chunks)
                progress_callback(percent, f"Clipping chunk {chunk_idx + 1}/{n_chunks}...")
        
        # Update metadata
        data.metadata["ClipRange"] = f"[{min_val:.0f}, {max_val:.0f}]"
        if "(Clipped)" not in data.metadata.get("Type", ""):
            data.metadata["Type"] = data.metadata.get("Type", "CT") + " (Clipped)"
        
        # Clear related caches
        self._clear_segmentation_cache()
        
        if progress_callback:
            progress_callback(100, "Clip complete")
        
        # Emit data changed signal
        self.data_changed.emit()
    
    def invert_data(self, chunk_size: int = 32,
                    progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[float, float, float, float]:
        """
        Invert volume values (for extracting pore surfaces instead of object surfaces).
        
        Args:
            chunk_size: Number of slices to process per chunk
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (data_min, data_max, invert_offset, new value range)
            
        Raises:
            ValueError: If no active data available
        """
        data = self.active_data
        if data is None or data.raw_data is None:
            raise ValueError("No active data to invert")
        
        raw = data.raw_data
        data_min = float(raw.min())
        data_max = float(raw.max())
        invert_offset = data_max + data_min
        n_slices = raw.shape[0]
        n_chunks = (n_slices + chunk_size - 1) // chunk_size
        
        if progress_callback:
            progress_callback(0, "Inverting volume...")
        
        # Chunked processing: new_val = max + min - val
        for i in range(0, n_slices, chunk_size):
            end = min(i + chunk_size, n_slices)
            raw[i:end] = invert_offset - raw[i:end]
            gc.collect()
            
            if progress_callback:
                chunk_idx = i // chunk_size
                percent = int(80 * (chunk_idx + 1) / n_chunks)
                progress_callback(percent, f"Inverting chunk {chunk_idx + 1}/{n_chunks}...")
        
        # Update metadata
        if "(Inverted)" not in data.metadata.get("Type", ""):
            data.metadata["Type"] = data.metadata.get("Type", "CT") + " (Inverted)"
        
        # Clear related caches
        self._clear_segmentation_cache()
        
        if progress_callback:
            progress_callback(100, "Invert complete")
        
        # Emit data changed signal
        self.data_changed.emit()
        
        return data_min, data_max, invert_offset
    
    def calculate_histogram(self, bins: int = 100, 
                            sample_step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram of active data with optional downsampling for performance.
        
        Args:
            bins: Number of histogram bins
            sample_step: Downsampling step for large volumes (applied in each dimension)
            
        Returns:
            Tuple of (histogram counts, bin edges)
        """
        data = self.active_data
        if data is None or data.raw_data is None:
            return np.array([]), np.array([])
        
        raw = data.raw_data
        
        # Downsample for performance if needed
        if raw.size > 10**6:
            sample = raw[::sample_step, ::sample_step, ::sample_step].flatten()
        else:
            sample = raw.flatten()
        
        hist, bin_edges = np.histogram(sample, bins=bins)
        return hist, bin_edges
    
    def get_data_range(self) -> Tuple[float, float]:
        """Get min/max values of active data."""
        data = self.active_data
        if data is None or data.raw_data is None:
            return (0.0, 1.0)
        
        return float(np.nanmin(data.raw_data)), float(np.nanmax(data.raw_data))

    # ==========================================
    # Cache Management
    # ==========================================
    
    def _clear_segmentation_cache(self):
        """Clear segmentation cache when data changes."""
        try:
            from data.disk_cache import clear_segmentation_cache
            clear_segmentation_cache()
        except Exception as e:
            print(f"[DataManager] Failed to clear segmentation cache: {e}")

    # ==========================================
    # Status Methods
    # ==========================================

    def get_current_state_info(self) -> str:
        """Returns a status string describing what data is available."""
        status = []
        if self.raw_ct_data:
            status.append("✔ Raw CT Data")
        else:
            status.append("✘ Raw CT Data")
            
        if self.roi_data:
            status.append("✔ ROI Extracted")

        if self.segmented_volume:
            status.append("✔ Segmented Void Volume")
        else:
            status.append("✘ Segmented Void Volume")

        if self.pnm_model:
            status.append("✔ PNM Mesh (Pores & Throats)")
        else:
            status.append("✘ PNM Mesh")

        return " | ".join(status)

    def has_raw(self) -> bool:
        return self.raw_ct_data is not None
    
    def has_active_data(self) -> bool:
        return self.active_data is not None

    def has_segmented(self) -> bool:
        return self.segmented_volume is not None
