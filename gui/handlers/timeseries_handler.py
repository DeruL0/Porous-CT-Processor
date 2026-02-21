from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core import VolumeData
from data import get_timeseries_pnm_cache
from loaders import AnnotationValidator, SmartDicomLoader, TimeSeriesDicomLoader
from loaders.dicom import LoadStrategy
from loaders.time_series import TimeSeriesOrderDialog
from processors import PNMTracker, PoreExtractionProcessor, PoreToSphereProcessor


class TimeSeriesLoadWorker(QThread):
    """Background worker for 4D CT series loading + annotation validation."""

    finished = pyqtSignal(object)  # list[VolumeData]
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        parent_folder: str,
        sort_mode: str,
        manual_order: Optional[List[str]],
        strategy: Optional[LoadStrategy],
    ):
        super().__init__()
        self.parent_folder = parent_folder
        self.sort_mode = sort_mode
        self.manual_order = manual_order
        self.strategy = strategy
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            loader = TimeSeriesDicomLoader(loader=SmartDicomLoader(strategy=self.strategy))

            def callback(percent: int, message: str) -> None:
                if self._cancelled:
                    raise InterruptedError("Operation cancelled by user.")
                self.progress.emit(max(0, min(95, int(percent))), message)

            volumes = loader.load_series(
                self.parent_folder,
                sort_mode=self.sort_mode,
                manual_order=self.manual_order,
                callback=callback,
            )

            if self._cancelled:
                self.cancelled.emit()
                return

            self.progress.emit(96, "Validating simulation annotations...")
            AnnotationValidator(strict=False).validate_series(parent_folder=self.parent_folder, volumes=volumes)
            self.progress.emit(100, "4D CT loading complete.")

            if self._cancelled:
                self.cancelled.emit()
                return
            self.finished.emit(volumes)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class TimeSeriesTrackWorker(QThread):
    """Background worker for 4D CT pore tracking + evaluation."""

    finished = pyqtSignal(object)  # dict result
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, volumes: List[VolumeData], threshold: int):
        super().__init__()
        self.volumes = volumes
        self.threshold = int(threshold)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise InterruptedError("Operation cancelled by user.")

    def _phase_callback(self, start_pct: int, end_pct: int, prefix: str = ""):
        span = max(1, end_pct - start_pct)

        def callback(percent: int, message: str) -> None:
            self._check_cancel()
            local = max(0, min(100, int(percent)))
            mapped = start_pct + int(span * local / 100)
            self.progress.emit(min(100, mapped), f"{prefix}{message}")

        return callback

    def run(self) -> None:
        try:
            self._check_cancel()
            tracker = PNMTracker()
            sphere_processor = PoreToSphereProcessor()
            total = len(self.volumes)

            reference_snapshot = None
            for i, volume in enumerate(self.volumes):
                self._check_cancel()
                step_start = int(80 * i / total)
                step_end = max(step_start + 1, int(80 * (i + 1) / total))
                self.progress.emit(step_start, f"Tracking timepoint {i + 1}/{total}...")

                compute_connections = i == 0
                snapshot = sphere_processor.extract_snapshot(
                    volume,
                    threshold=self.threshold,
                    time_index=i,
                    compute_connectivity=compute_connections,
                    callback=self._phase_callback(step_start, step_end, prefix=f"[t={i}] "),
                )

                if i == 0:
                    tracker.set_reference(snapshot)
                    reference_snapshot = snapshot
                else:
                    snapshot.connections = reference_snapshot.connections
                    tracker.track_snapshot(snapshot)

            self._check_cancel()
            self.progress.emit(85, "Generating reference PNM mesh...")
            reference_mesh = sphere_processor.process(
                self.volumes[0],
                threshold=self.threshold,
                callback=self._phase_callback(85, 88, prefix="[mesh] "),
            )
            pnm_result = tracker.get_results()

            self._check_cancel()
            self.progress.emit(88, "Evaluating against simulation labels...")
            try:
                tracker.evaluate_against_sim_annotations(self.volumes)
                pnm_result = tracker.get_results()
            except Exception as eval_exc:
                self.progress.emit(89, f"Simulation evaluation skipped: {eval_exc}")

            self._check_cancel()
            self.progress.emit(100, "Tracking complete.")
            self.finished.emit(
                {
                    "pnm_result": pnm_result,
                    "reference_mesh": reference_mesh,
                    "reference_snapshot": reference_snapshot,
                }
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


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

        self._volumes: List[VolumeData] = []
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache: Dict[int, VolumeData] = {}
        self._current_timepoint = 0

        self._sphere_processor = PoreToSphereProcessor()
        self._pore_processor = PoreExtractionProcessor()
        self._pnm_cache = get_timeseries_pnm_cache()

        self._load_worker: Optional[TimeSeriesLoadWorker] = None
        self._track_worker: Optional[TimeSeriesTrackWorker] = None
        self._progress_dialog = None

        if hasattr(self.analysis_panel, "pore_selected"):
            self.analysis_panel.pore_selected.connect(self._on_pore_selected)

    @property
    def has_volumes(self) -> bool:
        return bool(self._volumes)

    @property
    def is_busy(self) -> bool:
        workers = (self._load_worker, self._track_worker)
        return any(w is not None and w.isRunning() for w in workers)

    def _set_ui_busy(self, busy: bool) -> None:
        self.panel.setEnabled(not busy)
        self.control_panel.setEnabled(not busy)

    def _ensure_idle(self) -> bool:
        if self.is_busy:
            QMessageBox.warning(self.visualizer, "Busy", "A 4D CT task is currently running.")
            return False
        if hasattr(self.controller, "workflow_handler") and self.controller.workflow_handler.is_busy:
            QMessageBox.warning(self.visualizer, "Busy", "A workflow task is currently running.")
            return False
        return True

    def load_series(self, strategy=None):
        """Load 4D CT series workflow."""
        if not self._ensure_idle():
            return

        folder = QFileDialog.getExistingDirectory(
            self.visualizer,
            "Select 4D CT Parent Folder (containing t0, t1, t2... subfolders)",
        )
        if not folder:
            return

        if self._current_cache_key:
            print("[TimeseriesHandler] Existing PNM cache key present; new series will compute new key.")

        sort_mode = self.panel.sort_combo.currentText().lower()
        manual_order = None

        if sort_mode == "manual":
            probe_loader = TimeSeriesDicomLoader(loader=SmartDicomLoader(strategy=strategy))
            folder_list = probe_loader.get_folder_list(folder, sort_mode="alphabetical")
            if not folder_list:
                QMessageBox.warning(self.visualizer, "No Data", "No DICOM subfolders found in selected folder.")
                return

            manual_order = TimeSeriesOrderDialog.get_order(self.visualizer, folder_list)
            if manual_order is None:
                return

        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog("Loading 4D CT time series...")
        self._load_worker = TimeSeriesLoadWorker(
            parent_folder=folder,
            sort_mode=sort_mode,
            manual_order=manual_order,
            strategy=strategy,
        )

        def on_progress(percent: int, message: str):
            self._progress_dialog.setValue(percent)
            self._progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)

        def on_finished(volumes: List[VolumeData]):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None

            self._volumes = volumes
            if not self._volumes:
                self.controller._show_err("4D CT Loading Error", ValueError("No timepoints loaded"))
                self.visualizer.update_status("4D CT loading failed.")
                return

            self._initialize_series()
            annotation_msg = ""
            annotation_report = self._summarize_sim_annotation_validation()
            if annotation_report is not None:
                annotation_msg = (
                    f"\n\nAnnotation checks:\n"
                    f"- Errors: {annotation_report['error_count']}\n"
                    f"- Warnings: {annotation_report['warning_count']}"
                )

            self.controller._show_msg(
                "4D CT Loaded",
                f"Loaded {len(self._volumes)} timepoints.\n"
                f"Sorting: {sort_mode.title()}\n"
                f"Use timeline slider (left) to navigate volumes."
                f"{annotation_msg}",
            )
            self.visualizer.update_status(f"4D CT: {len(self._volumes)} timepoints loaded.")

        def on_cancelled():
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None
            self.visualizer.update_status("4D CT loading cancelled.")

        def on_error(message: str):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None
            self.controller._show_err("4D CT Loading Error", Exception(message))

        def on_cancel_requested():
            if self._load_worker is not None:
                self._load_worker.cancel()
                self.visualizer.update_status("Cancelling...")

        self._load_worker.progress.connect(on_progress)
        self._load_worker.finished.connect(on_finished)
        self._load_worker.cancelled.connect(on_cancelled)
        self._load_worker.error.connect(on_error)
        self._progress_dialog.canceled.connect(on_cancel_requested)
        self._load_worker.start()

    def _initialize_series(self):
        """Initialize UI and state after loading."""
        first_vol = self._volumes[0]
        self.data_manager.load_raw_data(first_vol)
        self.visualizer.set_data(first_vol)
        self.panel.set_threshold(-300)
        self.stats_panel.update_statistics(first_vol.metadata)

        names = [v.metadata.get("folder_name", f"t={i}") for i, v in enumerate(self._volumes)]
        self.control_panel.set_range(len(self._volumes), folder_names=names)

        self.analysis_panel.reset()
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache.clear()

    def _summarize_sim_annotation_validation(self):
        """Aggregate annotation validation results from loader metadata."""
        if not self._volumes:
            return None

        series_report = self._volumes[0].metadata.get("sim_annotation_series")
        if not isinstance(series_report, dict):
            return None
        validation = series_report.get("validation", {})
        return {
            "ok": bool(validation.get("ok", True)),
            "error_count": int(validation.get("error_count", 0)),
            "warning_count": int(validation.get("warning_count", 0)),
            "errors": list(validation.get("errors", [])),
            "warnings": list(validation.get("warnings", [])),
        }

    def track_pores(self):
        """Execute pore tracking workflow."""
        if not self._ensure_idle():
            return
        if not self._volumes:
            QMessageBox.warning(self.visualizer, "No 4D CT Data", "Please load a 4D CT series first.")
            return

        thresh = self.panel.get_threshold()
        cache_key = self._pnm_cache.generate_key(self._volumes, thresh)
        cached_data = self._pnm_cache.get(cache_key)
        if cached_data:
            self._pnm_result, self._reference_mesh, self._reference_snapshot = cached_data
            self._current_cache_key = cache_key

            self.analysis_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            self.stats_panel.update_statistics(self._reference_mesh.metadata)

            self._show_tracking_summary()
            self.visualizer.update_status("4D CT PNM loaded from cache.")
            return

        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog("Tracking pores across timepoints...")
        self._track_worker = TimeSeriesTrackWorker(self._volumes, thresh)

        def on_progress(percent: int, message: str):
            self._progress_dialog.setValue(percent)
            self._progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)

        def on_finished(result: Dict[str, object]):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._track_worker = None

            self._pnm_result = result["pnm_result"]
            self._reference_mesh = result["reference_mesh"]
            self._reference_snapshot = result["reference_snapshot"]

            self._pnm_cache.store(cache_key, self._pnm_result, self._reference_mesh, self._reference_snapshot)
            self._current_cache_key = cache_key

            self.analysis_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            self.stats_panel.update_statistics(self._reference_mesh.metadata)
            self._show_tracking_summary()
            self.visualizer.update_status("4D CT tracking complete.")

        def on_cancelled():
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._track_worker = None
            self.visualizer.update_status("4D CT tracking cancelled.")

        def on_error(message: str):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._track_worker = None
            self.controller._show_err("4D CT Tracking Error", Exception(message))

        def on_cancel_requested():
            if self._track_worker is not None:
                self._track_worker.cancel()
                self.visualizer.update_status("Cancelling...")

        self._track_worker.progress.connect(on_progress)
        self._track_worker.finished.connect(on_finished)
        self._track_worker.cancelled.connect(on_cancelled)
        self._track_worker.error.connect(on_error)
        self._progress_dialog.canceled.connect(on_cancel_requested)
        self._track_worker.start()

    def set_timepoint(self, index: int):
        """Handle timepoint change request."""
        if self.is_busy:
            return
        if not self._volumes or index >= len(self._volumes):
            return

        self._current_timepoint = index
        self.analysis_panel.set_timepoint(index)
        current_mode = self.visualizer.active_view_mode

        if current_mode == "mesh" and self._reference_mesh and self._pnm_result:
            current_snapshot = None
            if index < len(self._pnm_result.snapshots):
                current_snapshot = self._pnm_result.snapshots[index]

            mesh = self._sphere_processor.create_time_varying_mesh(
                self._reference_mesh,
                self._reference_snapshot,
                self._pnm_result.tracking,
                index,
                current_snapshot=current_snapshot,
            )
            self.visualizer.set_data(mesh, reset_camera=False)
            self.stats_panel.update_statistics(mesh.metadata)
            self.visualizer.update_status(f"Viewing PNM at t={index} (connectivity from t=0)")
            return

        if current_mode in ["volume", "slices", "iso"] and self.data_manager.has_segmented():
            if index in self._pores_cache:
                pores_data = self._pores_cache[index]
                self.visualizer.set_data(pores_data, reset_camera=False)
                self.stats_panel.update_statistics(pores_data.metadata)
                self.visualizer.update_status(f"Viewing pores at t={index} (cached)")
                return

            thresh = self.panel.get_threshold()
            try:
                pores_data = self._pore_processor.process(self._volumes[index], threshold=thresh)
                self._pores_cache[index] = pores_data
                self.visualizer.set_data(pores_data, reset_camera=False)
                self.stats_panel.update_statistics(pores_data.metadata)
                self.visualizer.update_status(f"Viewing pores at t={index}")
                return
            except Exception as exc:
                print(f"[TimeseriesHandler] Failed to extract pores for t={index}: {exc}")

        self.visualizer.set_data(self._volumes[index], reset_camera=False)
        self.stats_panel.update_statistics(self._volumes[index].metadata)
        self.visualizer.update_status(f"Viewing volume at t={index}")

    def _show_tracking_summary(self):
        summary = self._pnm_result.get_summary()
        eval_msg = ""
        eval_report = getattr(self._pnm_result.tracking, "evaluation", {})
        if isinstance(eval_report, dict) and eval_report.get("available"):
            overall = eval_report.get("overall", {})
            mean_voxel_iou = float(overall.get("mean_voxel_iou", 0.0))
            mean_instance_f1 = float(overall.get("mean_instance_f1", 0.0))
            mean_tracking_acc = float(overall.get("mean_tracking_accuracy", 0.0))
            eval_msg = (
                f"\n\nSimulation-label evaluation:\n"
                f"- Mean voxel IoU: {mean_voxel_iou:.1%}\n"
                f"- Mean instance F1: {mean_instance_f1:.1%}\n"
                f"- Mean tracking accuracy: {mean_tracking_acc:.1%}"
            )

        self.controller._show_msg(
            "4D CT Tracking Complete",
            f"Tracked {summary['reference_pores']} pores across {summary['num_timepoints']} timepoints.\n\n"
            f"- Active pores: {summary['active_pores']}\n"
            f"- Compressed pores: {summary['compressed_pores']}\n"
            f"- Avg. volume retention: {summary['avg_volume_retention']:.1%}\n\n"
            f"Use the timeline (left) to navigate time steps.\n"
            f"Use the analysis table (right) to see pore details."
            f"{eval_msg}",
        )

    def _on_pore_selected(self, pore_id: int):
        """Handle pore selection from analysis panel for highlighting."""
        current_highlight = getattr(self.visualizer, "_highlight_pore_id", None)

        if current_highlight == pore_id:
            self.visualizer.set_highlight_pore(None)
            self.visualizer.update_status("Cleared pore highlight")
        else:
            self.visualizer.set_highlight_pore(pore_id)
            self.visualizer.update_status(f"Highlighting pore {pore_id}")

        if self.visualizer.active_view_mode == "mesh":
            self.visualizer.render_mesh(reset_view=False)

    def cleanup(self):
        """Clean up resources and cache when closing or switching datasets."""
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.cancel()
            self._load_worker.wait(2000)
        if self._track_worker is not None and self._track_worker.isRunning():
            self._track_worker.cancel()
            self._track_worker.wait(2000)

        self._volumes.clear()
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache.clear()
        self._current_timepoint = 0

        if hasattr(self.visualizer, "clear_pnm_color_cache"):
            self.visualizer.clear_pnm_color_cache()
