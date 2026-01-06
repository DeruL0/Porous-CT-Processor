"""
ROI (Region of Interest) handler for volume visualization.
"""

from typing import Optional
import numpy as np
from core import VolumeData


class ROIHandler:
    """
    Handles ROI selection and extraction logic.
    Designed for composition with GUI/rendering classes.
    """

    def __init__(self, plotter, roi_panel, data_manager=None, status_callback=None):
        """
        Initialize ROI handler.
        
        Args:
            plotter: BackgroundPlotter instance
            roi_panel: ROIPanel for UI controls
            data_manager: Optional DataManager for data flow
            status_callback: Optional callback for status updates
        """
        self.plotter = plotter
        self.roi_panel = roi_panel
        self._data_manager = data_manager
        self._status_callback = status_callback
        
        # Reference to data (set externally)
        self.data: Optional[VolumeData] = None
        self.grid = None

    def update_status(self, message: str):
        """Update status via callback."""
        if self._status_callback:
            self._status_callback(message)
        else:
            print(f"[ROIHandler] {message}")

    def set_data(self, data: VolumeData, grid):
        """Set data reference for ROI extraction."""
        self.data = data
        self.grid = grid

    def on_roi_toggled(self, enabled: bool):
        """Handle ROI mode toggle - show/hide box widget."""
        if enabled:
            if self.grid is None:
                self.roi_panel.enable_checkbox.setChecked(False)
                return

            bounds = self.grid.bounds
            self.plotter.add_box_widget(
                callback=self.on_roi_bounds_changed,
                bounds=bounds,
                factor=1.0,
                rotation_enabled=False,
                color='cyan',
                use_planes=False
            )
            self.update_status("ROI mode: Drag the box to select region")
        else:
            self.plotter.clear_box_widgets()
            self.roi_panel.update_bounds(None)
            self.update_status("ROI mode disabled")

    def on_roi_bounds_changed(self, bounds):
        """Callback when user moves the ROI box widget."""
        if hasattr(bounds, 'bounds'):
            actual_bounds = bounds.bounds
        else:
            actual_bounds = bounds
        self.roi_panel.update_bounds(actual_bounds)

    def on_apply_roi(self, set_data_callback):
        """
        Extract sub-volume from ROI bounds and update data.
        
        Args:
            set_data_callback: Function to call with extracted VolumeData
        """
        roi_bounds = self.roi_panel.get_bounds()
        if roi_bounds is None or self.data is None:
            return

        try:
            extracted = self.extract_roi_subvolume(roi_bounds)
            if extracted is not None:
                if self._data_manager is not None:
                    self._data_manager.set_roi_data(extracted)

                set_data_callback(extracted)
                self.update_status(f"ROI applied: {extracted.raw_data.shape}")

                self.roi_panel.enable_checkbox.setChecked(False)
                self.plotter.clear_box_widgets()
        except Exception as e:
            print(f"[ROIHandler] Error applying: {e}")
            self.update_status("ROI extraction failed")

    def on_reset_roi(self, set_data_callback):
        """
        Reset ROI - clear box widget and reset to original data.
        
        Args:
            set_data_callback: Function to call with original VolumeData
        """
        self.plotter.clear_box_widgets()

        if self._data_manager is not None:
            self._data_manager.clear_roi()
            if self._data_manager.raw_ct_data is not None:
                set_data_callback(self._data_manager.raw_ct_data)

        self.update_status("ROI reset")

    def extract_roi_subvolume(self, bounds) -> Optional[VolumeData]:
        """Extract a sub-volume based on ROI bounds."""
        if self.data is None or self.data.raw_data is None:
            return None

        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin

        # AXIS MAPPING:
        # PyVista X -> numpy axis 0 (Z)
        # PyVista Y -> numpy axis 1 (Y)
        # PyVista Z -> numpy axis 2 (X)

        i_start = max(0, int((bounds[0] - origin[0]) / spacing[0]))
        i_end = min(raw.shape[0], int((bounds[1] - origin[0]) / spacing[0]))

        j_start = max(0, int((bounds[2] - origin[1]) / spacing[1]))
        j_end = min(raw.shape[1], int((bounds[3] - origin[1]) / spacing[1]))

        k_start = max(0, int((bounds[4] - origin[2]) / spacing[2]))
        k_end = min(raw.shape[2], int((bounds[5] - origin[2]) / spacing[2]))

        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end]

        if sub_data.size == 0:
            return None

        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )

        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Extract ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds

        return VolumeData(
            raw_data=sub_data,
            spacing=spacing,
            origin=new_origin,
            metadata=new_metadata
        )
