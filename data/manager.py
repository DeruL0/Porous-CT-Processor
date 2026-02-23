"""
Scientific data manager for porous media analysis workflow.

Responsibilities:
- Manages runtime data state (raw, segmented, PNM, ROI)
- Provides data manipulation operations (clip, invert, histogram)
- Coordinates cache lifecycle
- Emits signals on data changes for UI synchronization
"""

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
        # Clear all previous data to free memory.
        # Explicit reference drops trigger CPython's ref-count destructor
        # immediately 鈥?no gc.collect() required.
        self.raw_ct_data       = None
        self.segmented_volume  = None
        self.pnm_model         = None
        self.roi_data          = None

        # Clear segmentation cache
        self._clear_segmentation_cache()

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
    # Atomic Volume Operations
    # ==========================================

    @staticmethod
    def clip_volume_inplace(volume: VolumeData, min_val: float, max_val: float) -> None:
        """
        Clip a volume in-place to [min_val, max_val].

        Shared by 3D and 4D flows to keep transform behavior identical.
        """
        if volume is None or volume.raw_data is None:
            raise ValueError("No volume data to clip")

        np.clip(volume.raw_data, min_val, max_val, out=volume.raw_data)
        volume.metadata["ClipRange"] = f"[{min_val:.0f}, {max_val:.0f}]"
        if "(Clipped)" not in volume.metadata.get("Type", ""):
            volume.metadata["Type"] = volume.metadata.get("Type", "CT") + " (Clipped)"

    @staticmethod
    def invert_volume_inplace(volume: VolumeData) -> Tuple[float, float, float]:
        """
        Invert one volume in-place: ``new = (max + min) - old``.

        Returns:
            Tuple[data_min, data_max, invert_offset]
        """
        if volume is None or volume.raw_data is None:
            raise ValueError("No volume data to invert")

        raw = volume.raw_data
        data_min = float(raw.min())
        data_max = float(raw.max())
        invert_offset = data_max + data_min
        np.subtract(invert_offset, raw, out=raw)

        if "(Inverted)" not in volume.metadata.get("Type", ""):
            volume.metadata["Type"] = volume.metadata.get("Type", "CT") + " (Inverted)"
        return data_min, data_max, invert_offset

    # ==========================================
    # Data Manipulation Methods
    # ==========================================
    
    def clip_data(self, min_val: float, max_val: float,
                  chunk_size: int = 32,
                  progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """
        Permanently clip active data to [min_val, max_val] in-place.

        Uses ``np.clip(out=raw)`` for a fully vectorised, zero-extra-copy
        operation.  The ``chunk_size`` parameter is kept for API compatibility
        but is no longer used (vectorised numpy is both faster and creates no
        additional peak-memory spike for a simple clip).

        Args:
            min_val: Minimum value to clip to
            max_val: Maximum value to clip to
            chunk_size: Retained for backward-compatibility; ignored.
            progress_callback: Optional callback for progress updates (percent, message)

        Raises:
            ValueError: If no active data available
        """
        data = self.active_data
        if data is None or data.raw_data is None:
            raise ValueError("No active data to clip")

        if progress_callback:
            progress_callback(0, "Clipping data...")
        self.clip_volume_inplace(data, min_val=min_val, max_val=max_val)
        # Clear related caches
        self._clear_segmentation_cache()

        if progress_callback:
            progress_callback(100, "Clip complete")

        # Emit data changed signal
        self.data_changed.emit()
    
    def invert_data(self, chunk_size: int = 32,
                    progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[float, float, float]:
        """
        Invert volume values in-place: ``new_val = (max + min) - val``.

        Uses a fully vectorised numpy operation (``np.subtract(offset, raw,
        out=raw)``) instead of a chunked loop with gc.collect().  This avoids
        Stop-The-World pauses while being 2鈥?0脳 faster than the chunked path.

        Args:
            chunk_size: Retained for backward-compatibility; ignored.
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (data_min, data_max, invert_offset)

        Raises:
            ValueError: If no active data available
        """
        data = self.active_data
        if data is None or data.raw_data is None:
            raise ValueError("No active data to invert")
        if progress_callback:
            progress_callback(0, "Inverting volume...")

        data_min, data_max, invert_offset = self.invert_volume_inplace(data)
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
            status.append("鉁?Raw CT Data")
        else:
            status.append("鉁?Raw CT Data")
            
        if self.roi_data:
            status.append("鉁?ROI Extracted")

        if self.segmented_volume:
            status.append("鉁?Segmented Void Volume")
        else:
            status.append("鉁?Segmented Void Volume")

        if self.pnm_model:
            status.append("鉁?PNM Mesh (Pores & Throats)")
        else:
            status.append("鉁?PNM Mesh")

        return " | ".join(status)

    def has_raw(self) -> bool:
        return self.raw_ct_data is not None
    
    def has_active_data(self) -> bool:
        return self.active_data is not None

    def has_segmented(self) -> bool:
        return self.segmented_volume is not None

