from typing import List, Optional
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject

from core import VolumeData
from loaders import TimeSeriesDicomLoader
from loaders.time_series import TimeSeriesOrderDialog
from processors import PNMTracker, PoreToSphereProcessor

class TimeseriesHandler(QObject):
    """
    Handler for 4D CT Time Series operations.
    Manages loading, tracking, and visualization of time-series data.
    """
    
    def __init__(self, main_controller):
        super().__init__()
        self.controller = main_controller
        self.visualizer = main_controller.visualizer
        self.panel = main_controller.panel
        self.timeseries_panel = main_controller.timeseries_panel
        self.data_manager = main_controller.data_manager
        
        # State
        self._volumes: List[VolumeData] = []
        self._pnm_result = None
        self._reference_mesh = None    # t=0 PNM mesh
        self._reference_snapshot = None # t=0 PNM snapshot
        
        # Helper processor for on-demand mesh generation
        self._sphere_processor = PoreToSphereProcessor()

    @property
    def has_volumes(self) -> bool:
        return bool(self._volumes)

    def load_series(self):
        """Load 4D CT series workflow."""
        folder = QFileDialog.getExistingDirectory(
            self.visualizer, 
            "Select 4D CT Parent Folder (containing t0, t1, t2... subfolders)"
        )
        if not folder:
            return
        
        sort_mode = self.panel.get_sort_mode()
        loader = TimeSeriesDicomLoader()
        manual_order = None
        
        # Manual sorting dialog
        if sort_mode == 'manual':
            folder_list = loader.get_folder_list(folder, sort_mode='alphabetical')
            if not folder_list:
                QMessageBox.warning(self.visualizer, "No Data", 
                                   "No DICOM subfolders found in selected folder.")
                return
            
            manual_order = TimeSeriesOrderDialog.get_order(self.visualizer, folder_list)
            if manual_order is None:
                return
        
        # Execute loading
        progress = self.controller._create_progress_dialog("Loading 4D CT time series...")
        callback = self.controller._make_progress_callback(progress)
        
        try:
            self.visualizer.update_status("Loading 4D CT time series...")
            
            self._volumes = loader.load_series(
                folder, 
                sort_mode=sort_mode,
                manual_order=manual_order,
                callback=callback
            )
            
            if not self._volumes:
                raise ValueError("No timepoints loaded")
            
            self._initialize_series()
            
            self.controller._show_msg(
                "4D CT Loaded", 
                f"Loaded {len(self._volumes)} timepoints.\n"
                f"Sorting: {sort_mode.title()}\n"
                f"Use timeline slider to navigate volumes.\n"
                f"Use 'Track 4D Pores' to generate PNM and track changes."
            )
            self.visualizer.update_status(f"4D CT: {len(self._volumes)} timepoints loaded.")
            
        except Exception as e:
            self.controller._show_err("4D CT Loading Error", e)
        finally:
            progress.close()

    def _initialize_series(self):
        """Initialize UI and state after loading."""
        # Load t=0
        first_vol = self._volumes[0]
        self.data_manager.load_raw_data(first_vol)
        self.visualizer.set_data(first_vol)
        self.panel.set_threshold(-300)
        
        # Setup timeline
        names = [v.metadata.get('folder_name', f't={i}') for i, v in enumerate(self._volumes)]
        self.timeseries_panel.set_volume_only_mode(len(self._volumes), folder_names=names)
        
        # Reset tracking state
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None

    def track_pores(self):
        """Execute pore tracking workflow."""
        if not self._volumes:
            QMessageBox.warning(self.visualizer, "No 4D CT Data", 
                               "Please load a 4D CT series first.")
            return
        
        thresh = self.panel.get_threshold()
        progress = self.controller._create_progress_dialog("Tracking pores across timepoints...")
        
        try:
            tracker = PNMTracker()
            total = len(self._volumes)
            
            # Phase 1: Tracking
            for i, volume in enumerate(self._volumes):
                progress.setValue(int(80 * i / total))
                progress.setLabelText(f"Tracking timepoint {i+1}/{total}...")
                self.controller.app.processEvents()
                
                if progress.wasCanceled(): raise InterruptedError()
                
                snapshot = self._sphere_processor.extract_snapshot(volume, threshold=thresh, time_index=i)
                
                if i == 0:
                    tracker.set_reference(snapshot)
                    self._reference_snapshot = snapshot
                else:
                    tracker.track_snapshot(snapshot)
            
            # Phase 2: Reference Mesh
            progress.setValue(85)
            progress.setLabelText("Generating reference PNM mesh...")
            self.controller.app.processEvents()
            
            self._reference_mesh = self._sphere_processor.process(self._volumes[0], threshold=thresh)
            self._pnm_result = tracker.get_results()
            
            # Update UI
            progress.setValue(95)
            self.timeseries_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            
            self._show_tracking_summary()
            
        except Exception as e:
            self.controller._show_err("4D CT Tracking Error", e)
        finally:
            progress.close()

    def set_timepoint(self, index: int):
        """Handle timepoint change request."""
        if not self._volumes or index >= len(self._volumes):
            return
            
        # If we have a full tracking result, show PNM mesh
        if self._reference_mesh and self._pnm_result:
            if index == 0:
                self.visualizer.set_data(self._reference_mesh, reset_camera=False)
            else:
                # Find current snapshot if available
                current_snapshot = None
                if index < len(self._pnm_result.snapshots):
                    current_snapshot = self._pnm_result.snapshots[index]
                    
                mesh = self._sphere_processor.create_time_varying_mesh(
                    self._reference_mesh,
                    self._reference_snapshot,
                    self._pnm_result.tracking,
                    index,
                    current_snapshot=current_snapshot
                )
                self.visualizer.set_data(mesh, reset_camera=False)
            self.visualizer.update_status(f"Viewing PNM at t={index} (connectivity from t=0)")
        else:
            # Otherwise just show raw volume
            self.visualizer.set_data(self._volumes[index], reset_camera=False)
            self.visualizer.update_status(f"Viewing volume at t={index}")

    def _show_tracking_summary(self):
        summary = self._pnm_result.get_summary()
        self.controller._show_msg(
            "4D CT Tracking Complete",
            f"Tracked {summary['reference_pores']} pores across {summary['num_timepoints']} timepoints.\n\n"
            f"• Active pores: {summary['active_pores']}\n"
            f"• Compressed pores: {summary['compressed_pores']}\n"
            f"• Avg. volume retention: {summary['avg_volume_retention']:.1%}\n\n"
            f"Connectivity is preserved from t=0.\n"
            f"Use the timeline to view size changes over time."
        )
