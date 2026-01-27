from typing import List, Optional, Dict
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject

from core import VolumeData
from loaders import TimeSeriesDicomLoader
from loaders.time_series import TimeSeriesOrderDialog
from processors import PNMTracker, PoreToSphereProcessor, PoreExtractionProcessor
from data import get_timeseries_pnm_cache

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
        self.control_panel = main_controller.timeseries_control
        self.analysis_panel = main_controller.tracking_analysis
        self.stats_panel = main_controller.stats_panel
        self.data_manager = main_controller.data_manager

        
        # State
        self._volumes: List[VolumeData] = []
        self._pnm_result = None
        self._reference_mesh = None    # t=0 PNM mesh
        self._reference_snapshot = None # t=0 PNM snapshot
        self._current_cache_key = None  # Current cache key for PNM data
        self._pores_cache: Dict[int, VolumeData] = {}  # Cache for extracted pores at each timepoint
        
        # Helper processor for on-demand mesh generation
        self._sphere_processor = PoreToSphereProcessor()
        self._pore_processor = PoreExtractionProcessor()
        
        # Cache manager
        self._pnm_cache = get_timeseries_pnm_cache()

    @property
    def has_volumes(self) -> bool:
        return bool(self._volumes)

    def load_series(self, strategy=None):
        """Load 4D CT series workflow."""
        from loaders import SmartDicomLoader
        folder = QFileDialog.getExistingDirectory(
            self.visualizer, 
            "Select 4D CT Parent Folder (containing t0, t1, t2... subfolders)"
        )
        if not folder:
            return
        
        # Clear old PNM cache when loading new series
        if self._current_cache_key:
            print("[TimeseriesHandler] Clearing old PNM cache before loading new series")
            # Keep the cache for potential reuse if user switches back
            # Only clear if it's a different dataset
        
        sort_mode = self.panel.sort_combo.currentText().lower()
        
        # Use existing loaders by passing a SmartDicomLoader with desired strategy
        inner_loader = SmartDicomLoader(strategy=strategy)
        loader = TimeSeriesDicomLoader(loader=inner_loader)
        
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
                f"Use timeline slider (left) to navigate volumes."
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
        self.stats_panel.update_statistics(first_vol.metadata)

        
        # Setup timeline (left side)
        names = [v.metadata.get('folder_name', f't={i}') for i, v in enumerate(self._volumes)]
        self.control_panel.set_range(len(self._volumes), folder_names=names)
        
        # Reset analysis (right side)
        self.analysis_panel.reset()
        
        # Reset tracking state
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache.clear()  # Clear pores cache when loading new series

    def track_pores(self):
        """Execute pore tracking workflow."""
        if not self._volumes:
            QMessageBox.warning(self.visualizer, "No 4D CT Data", 
                               "Please load a 4D CT series first.")
            return
        
        thresh = self.panel.get_threshold()
        
        # Generate cache key for this configuration
        cache_key = self._pnm_cache.generate_key(self._volumes, thresh)
        
        # Check if we have cached results
        cached_data = self._pnm_cache.get(cache_key)
        if cached_data:
            self._pnm_result, self._reference_mesh, self._reference_snapshot = cached_data
            self._current_cache_key = cache_key
            
            # Update UI
            self.analysis_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            self.stats_panel.update_statistics(self._reference_mesh.metadata)
            
            self._show_tracking_summary()
            self.visualizer.update_status("4D CT PNM loaded from cache.")
            return
        
        # No cache, proceed with tracking
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
                
                # Only compute connectivity for t=0 (reference frame)
                # Subsequent frames inherit connectivity from reference
                compute_connections = (i == 0)
                snapshot = self._sphere_processor.extract_snapshot(
                    volume, threshold=thresh, time_index=i, 
                    compute_connectivity=compute_connections
                )
                
                if i == 0:
                    tracker.set_reference(snapshot)
                    self._reference_snapshot = snapshot
                else:
                    # Inherit connections from reference
                    snapshot.connections = self._reference_snapshot.connections
                    tracker.track_snapshot(snapshot)
            
            # Phase 2: Reference Mesh
            progress.setValue(85)
            progress.setLabelText("Generating reference PNM mesh...")
            self.controller.app.processEvents()
            
            self._reference_mesh = self._sphere_processor.process(self._volumes[0], threshold=thresh)
            self._pnm_result = tracker.get_results()
            
            # Store in cache
            progress.setValue(90)
            progress.setLabelText("Caching PNM results...")
            self.controller.app.processEvents()
            
            self._pnm_cache.store(cache_key, self._pnm_result, 
                                 self._reference_mesh, self._reference_snapshot)
            self._current_cache_key = cache_key
            
            # Update UI
            progress.setValue(95)
            self.analysis_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            self.stats_panel.update_statistics(self._reference_mesh.metadata)
            
            self._show_tracking_summary()

            
        except Exception as e:
            self.controller._show_err("4D CT Tracking Error", e)
        finally:
            progress.close()

    def set_timepoint(self, index: int):
        """Handle timepoint change request."""
        if not self._volumes or index >= len(self._volumes):
            return
        
        # Update analysis panel view
        self.analysis_panel.set_timepoint(index)
        
        # Check current view mode to maintain user's preference
        current_mode = self.visualizer.active_view_mode
        
        # If user is currently viewing PNM mesh and we have tracking data, update PNM view
        if current_mode == 'mesh' and self._reference_mesh and self._pnm_result:
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
            self.stats_panel.update_statistics(mesh.metadata)
            self.visualizer.update_status(f"Viewing PNM at t={index} (connectivity from t=0)")
        
        # If user is viewing extracted pores (segmented data), show pores for this timepoint
        elif current_mode in ['volume', 'slices', 'iso'] and self.data_manager.has_segmented():
            # Check if we have cached pores for this timepoint
            if index in self._pores_cache:
                pores_data = self._pores_cache[index]
                self.visualizer.set_data(pores_data, reset_camera=False)
                self.stats_panel.update_statistics(pores_data.metadata)
                self.visualizer.update_status(f"Viewing pores at t={index} (cached)")
            else:
                # Extract pores for this timepoint
                thresh = self.panel.get_threshold()
                try:
                    pores_data = self._pore_processor.process(
                        self._volumes[index], 
                        threshold=thresh
                    )
                    # Cache the result
                    self._pores_cache[index] = pores_data
                    
                    self.visualizer.set_data(pores_data, reset_camera=False)
                    self.stats_panel.update_statistics(pores_data.metadata)
                    self.visualizer.update_status(f"Viewing pores at t={index}")
                except Exception as e:
                    print(f"[TimeseriesHandler] Failed to extract pores for t={index}: {e}")
                    # Fallback to raw volume
                    self.visualizer.set_data(self._volumes[index], reset_camera=False)
                    self.stats_panel.update_statistics(self._volumes[index].metadata)
                    self.visualizer.update_status(f"Viewing volume at t={index}")
        
        else:
            # Otherwise show raw volume (user is in volume/slices/iso mode without pores data)
            self.visualizer.set_data(self._volumes[index], reset_camera=False)
            self.stats_panel.update_statistics(self._volumes[index].metadata)
            self.visualizer.update_status(f"Viewing volume at t={index}")


    def _show_tracking_summary(self):
        summary = self._pnm_result.get_summary()
        self.controller._show_msg(
            "4D CT Tracking Complete",
            f"Tracked {summary['reference_pores']} pores across {summary['num_timepoints']} timepoints.\n\n"
            f"• Active pores: {summary['active_pores']}\n"
            f"• Compressed pores: {summary['compressed_pores']}\n"
            f"• Avg. volume retention: {summary['avg_volume_retention']:.1%}\n\n"
            f"Use the timeline (left) to navigate time steps.\n"
            f"Use the analysis table (right) to see pore details."
        )

    def cleanup(self):
        """Clean up resources and cache when closing or switching datasets."""
        # Note: We don't clear all cache here, as it might be useful for re-opening
        # Only clear the current session's state
        self._volumes.clear()
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache.clear()
        
        # Optionally clear all PNM cache on explicit cleanup
        # Uncomment if you want to clear cache on every cleanup:
        # self._pnm_cache.clear()
