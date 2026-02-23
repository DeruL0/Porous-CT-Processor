from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core import VolumeData
from data import (
    ScientificDataManager,
    clear_segmentation_cache,
    clear_timeseries_pnm_cache,
    get_timeseries_pnm_cache,
)
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
            resolved_threshold = int(self.threshold)
            threshold_autofixed = False

            reference_snapshot = None
            for i, volume in enumerate(self.volumes):
                self._check_cancel()
                step_start = int(80 * i / total)
                step_end = max(step_start + 1, int(80 * (i + 1) / total))
                self.progress.emit(step_start, f"Tracking timepoint {i + 1}/{total}...")

                compute_connections = i == 0
                snapshot = sphere_processor.extract_snapshot(
                    volume,
                    threshold=resolved_threshold,
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
                threshold=resolved_threshold,
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
                    "resolved_threshold": int(resolved_threshold),
                    "threshold_autofixed": bool(threshold_autofixed),
                }
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class TimepointSegmentationWorker(QThread):
    """Lazy segmentation worker for one timepoint (used by segmented overlay)."""

    finished = pyqtSignal(object)  # dict result
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, volume: VolumeData, threshold: int, timepoint: int):
        super().__init__()
        self.volume = volume
        self.threshold = int(threshold)
        self.timepoint = int(timepoint)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            processor = PoreExtractionProcessor()

            def callback(percent: int, message: str) -> None:
                if self._cancelled:
                    raise InterruptedError("Operation cancelled by user.")
                self.progress.emit(
                    max(0, min(100, int(percent))),
                    f"[overlay t={self.timepoint}] {message}",
                )

            segmented = processor.process(
                self.volume,
                threshold=self.threshold,
                callback=callback,
            )

            if self._cancelled:
                self.cancelled.emit()
                return

            self.finished.emit(
                {
                    "timepoint": self.timepoint,
                    "threshold": self.threshold,
                    "segmented": segmented,
                }
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class TimeSeriesTransformWorker(QThread):
    """Background worker for clip/invert operations over all timepoints."""

    finished = pyqtSignal(object)  # dict result
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        volumes: List[VolumeData],
        mode: Literal["clip", "invert"],
        current_index: int,
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        super().__init__()
        self.volumes = volumes
        self.mode = mode
        self.current_index = int(current_index)
        self.min_val = min_val
        self.max_val = max_val
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise InterruptedError("Operation cancelled by user.")

    def _build_summary(self, invert_offset: Optional[float] = None) -> Dict[str, Any]:
        if not self.volumes:
            return {
                "mode": self.mode,
                "current_data_min": 0.0,
                "current_data_max": 0.0,
                "invert_offset": 0.0,
            }

        idx = min(max(self.current_index, 0), len(self.volumes) - 1)
        current = self.volumes[idx]
        if current.raw_data is None:
            current_min, current_max = 0.0, 0.0
        else:
            current_min = float(np.nanmin(current.raw_data))
            current_max = float(np.nanmax(current.raw_data))

        summary = {
            "mode": self.mode,
            "current_data_min": current_min,
            "current_data_max": current_max,
        }
        if self.mode == "invert":
            summary["invert_offset"] = (
                float(invert_offset) if invert_offset is not None else float(current_min + current_max)
            )
        return summary

    def run(self) -> None:
        try:
            total = len(self.volumes)
            if total == 0:
                raise ValueError("No 4D CT data loaded.")

            if self.mode == "clip":
                if self.min_val is None or self.max_val is None:
                    raise ValueError("Clip mode requires min_val and max_val.")

            current_invert_offset: Optional[float] = None
            for i, volume in enumerate(self.volumes):
                self._check_cancel()
                start_pct = int(100 * i / total)

                if self.mode == "clip":
                    self.progress.emit(
                        start_pct,
                        f"Applying 4D clip ({i + 1}/{total})...",
                    )
                    ScientificDataManager.clip_volume_inplace(
                        volume,
                        min_val=float(self.min_val),
                        max_val=float(self.max_val),
                    )
                elif self.mode == "invert":
                    self.progress.emit(
                        start_pct,
                        f"Inverting 4D volume ({i + 1}/{total})...",
                    )
                    _dmin, _dmax, invert_offset = ScientificDataManager.invert_volume_inplace(volume)
                    if i == self.current_index:
                        current_invert_offset = invert_offset
                else:
                    raise ValueError(f"Unsupported transform mode: {self.mode}")

                end_pct = int(100 * (i + 1) / total)
                self.progress.emit(end_pct, f"{self.mode.capitalize()} complete for t={i}.")

            summary = self._build_summary(invert_offset=current_invert_offset)
            self.progress.emit(100, f"4D {self.mode} complete.")
            self.finished.emit(summary)
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
        self._pores_cache: Dict[Tuple[int, int], VolumeData] = {}
        self._current_timepoint = 0
        self._overlay_seg_workers: Dict[Tuple[int, int], TimepointSegmentationWorker] = {}
        self._overlay_seg_callbacks: Dict[Tuple[int, int], List[Callable[[], None]]] = {}

        self._sphere_processor = PoreToSphereProcessor()
        self._pore_processor = PoreExtractionProcessor()
        self._pnm_cache = get_timeseries_pnm_cache()

        self._load_worker: Optional[TimeSeriesLoadWorker] = None
        self._track_worker: Optional[TimeSeriesTrackWorker] = None
        self._transform_worker: Optional[TimeSeriesTransformWorker] = None
        self._progress_dialog = None

        if hasattr(self.analysis_panel, "pore_selected"):
            self.analysis_panel.pore_selected.connect(self._on_pore_selected)

    @property
    def has_volumes(self) -> bool:
        return bool(self._volumes)

    @property
    def is_busy(self) -> bool:
        workers = (self._load_worker, self._track_worker, self._transform_worker)
        return any(w is not None and w.isRunning() for w in workers)

    @staticmethod
    def _pore_cache_key(timepoint: int, threshold: int) -> Tuple[int, int]:
        return int(timepoint), int(threshold)

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

    def _close_progress_dialog(self) -> None:
        """Close and clear progress dialog safely."""
        dialog = self._progress_dialog
        self._progress_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass

    def get_current_raw_volume(self) -> Optional[VolumeData]:
        """Return currently selected raw volume in 4D mode."""
        if not self._volumes:
            return None
        if self._current_timepoint < 0 or self._current_timepoint >= len(self._volumes):
            return None
        return self._volumes[self._current_timepoint]

    def cache_segmented_for_current(self, segmented: VolumeData) -> None:
        """
        Cache segmented data for current timepoint and threshold.
        """
        if segmented is None:
            return
        threshold = int(self.panel.get_threshold())
        key = self._pore_cache_key(self._current_timepoint, threshold)
        self._pores_cache[key] = segmented

    def get_or_request_segmented_overlay(
        self,
        threshold: int,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> Optional[VolumeData]:
        """
        Return segmented volume for current timepoint if cached; otherwise
        start lazy background extraction and return None.
        """
        if not self._volumes:
            return None

        threshold = int(threshold)
        timepoint = int(self._current_timepoint)
        key = self._pore_cache_key(timepoint, threshold)
        cached = self._pores_cache.get(key)
        if cached is not None:
            return cached

        if on_ready is not None:
            callbacks = self._overlay_seg_callbacks.setdefault(key, [])
            callbacks.append(on_ready)

        if key in self._overlay_seg_workers:
            return None

        worker = TimepointSegmentationWorker(
            volume=self._volumes[timepoint],
            threshold=threshold,
            timepoint=timepoint,
        )
        self._overlay_seg_workers[key] = worker

        def _cleanup() -> List[Callable[[], None]]:
            self._overlay_seg_workers.pop(key, None)
            return self._overlay_seg_callbacks.pop(key, [])

        def on_progress(_percent: int, message: str):
            self.visualizer.update_status(message)

        def on_finished(result: Dict[str, object]):
            callbacks = _cleanup()
            segmented = result.get("segmented")
            if isinstance(segmented, VolumeData):
                self._pores_cache[key] = segmented
            for cb in callbacks:
                try:
                    cb()
                except Exception:
                    pass

        def on_cancelled():
            _cleanup()

        def on_error(message: str):
            _cleanup()
            self.visualizer.update_status(f"Segmented overlay failed: {message}")

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.cancelled.connect(on_cancelled)
        worker.error.connect(on_error)
        worker.start()
        return None

    def _cancel_overlay_seg_workers(self) -> None:
        for worker in list(self._overlay_seg_workers.values()):
            try:
                worker.cancel()
                if worker.isRunning():
                    worker.wait(1000)
            except Exception:
                pass
        self._overlay_seg_workers.clear()
        self._overlay_seg_callbacks.clear()

    def _build_transform_summary(self, mode: Literal["clip", "invert"], cancelled: bool) -> Dict[str, Any]:
        current = self.get_current_raw_volume()
        if current is None or current.raw_data is None:
            payload: Dict[str, Any] = {
                "mode": mode,
                "cancelled": cancelled,
                "current_data_min": 0.0,
                "current_data_max": 0.0,
            }
            if mode == "invert":
                payload["invert_offset"] = 0.0
            return payload

        current_min = float(np.nanmin(current.raw_data))
        current_max = float(np.nanmax(current.raw_data))
        payload = {
            "mode": mode,
            "cancelled": cancelled,
            "current_data_min": current_min,
            "current_data_max": current_max,
        }
        if mode == "invert":
            payload["invert_offset"] = current_min + current_max
        return payload

    def _invalidate_processed_states(self) -> None:
        """Invalidate segmentation/tracking products after raw volume transform."""
        self._cancel_overlay_seg_workers()
        self._pores_cache.clear()
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self.data_manager.segmented_volume = None
        self.data_manager.pnm_model = None
        self.analysis_panel.reset()
        try:
            clear_segmentation_cache()
        except Exception:
            pass
        try:
            clear_timeseries_pnm_cache()
        except Exception:
            pass

    def run_series_transform(
        self,
        mode: Literal["clip", "invert"],
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Run clip/invert over the entire 4D sequence in background thread."""
        if not self._ensure_idle():
            return
        if not self._volumes:
            QMessageBox.warning(self.visualizer, "No 4D CT Data", "Please load a 4D CT series first.")
            return
        if mode not in {"clip", "invert"}:
            raise ValueError(f"Unsupported transform mode: {mode}")
        if mode == "clip" and (min_val is None or max_val is None):
            raise ValueError("clip mode requires min_val/max_val")

        # Avoid races between lazy segmentation and in-place raw-data transforms.
        self._cancel_overlay_seg_workers()

        self._set_ui_busy(True)
        title = "Applying 4D range clip..." if mode == "clip" else "Inverting 4D series..."
        self._progress_dialog = self.controller._create_progress_dialog(title)
        self._transform_worker = TimeSeriesTransformWorker(
            self._volumes,
            mode=mode,
            current_index=self._current_timepoint,
            min_val=min_val,
            max_val=max_val,
        )

        def finalize():
            self._close_progress_dialog()
            self._set_ui_busy(False)
            self._transform_worker = None

        def on_progress(percent: int, message: str):
            dialog = self._progress_dialog
            if dialog is not None:
                try:
                    dialog.setValue(percent)
                    dialog.setLabelText(message)
                except (RuntimeError, AttributeError):
                    pass
            self.visualizer.update_status(message)

        def on_finished(result: Dict[str, object]):
            finalize()
            self._invalidate_processed_states()
            self.set_timepoint(self._current_timepoint)
            payload = dict(result)
            payload["cancelled"] = False
            if completion_callback is not None:
                completion_callback(payload)

        def on_cancelled():
            finalize()
            self._invalidate_processed_states()
            self.set_timepoint(self._current_timepoint)
            payload = self._build_transform_summary(mode=mode, cancelled=True)
            if completion_callback is not None:
                completion_callback(payload)

        def on_error(message: str):
            finalize()
            self.controller._show_err(f"4D {mode.title()} Error", Exception(message))

        def on_cancel_requested():
            if self._transform_worker is not None:
                self._transform_worker.cancel()
                self.visualizer.update_status("Cancelling...")

        self._transform_worker.progress.connect(on_progress)
        self._transform_worker.finished.connect(on_finished)
        self._transform_worker.cancelled.connect(on_cancelled)
        self._transform_worker.error.connect(on_error)
        self._progress_dialog.canceled.connect(on_cancel_requested)
        self._transform_worker.start()

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
            dialog = self._progress_dialog
            if dialog is not None:
                try:
                    dialog.setValue(percent)
                    dialog.setLabelText(message)
                except (RuntimeError, AttributeError):
                    pass
            self.visualizer.update_status(message)

        def on_finished(volumes: List[VolumeData]):
            self._close_progress_dialog()
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
            self._close_progress_dialog()
            self._set_ui_busy(False)
            self._load_worker = None
            self.visualizer.update_status("4D CT loading cancelled.")

        def on_error(message: str):
            self._close_progress_dialog()
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
        self._cancel_overlay_seg_workers()
        self._current_timepoint = 0

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
            dialog = self._progress_dialog
            if dialog is not None:
                try:
                    dialog.setValue(percent)
                    dialog.setLabelText(message)
                except (RuntimeError, AttributeError):
                    pass
            self.visualizer.update_status(message)

        def on_finished(result: Dict[str, object]):
            self._close_progress_dialog()
            self._set_ui_busy(False)
            self._track_worker = None

            self._pnm_result = result["pnm_result"]
            self._reference_mesh = result["reference_mesh"]
            self._reference_snapshot = result["reference_snapshot"]
            resolved_threshold = int(result.get("resolved_threshold", thresh))
            threshold_autofixed = bool(result.get("threshold_autofixed", False))
            effective_cache_key = self._pnm_cache.generate_key(self._volumes, resolved_threshold)
            self._pnm_cache.store(
                effective_cache_key,
                self._pnm_result,
                self._reference_mesh,
                self._reference_snapshot,
            )
            self._current_cache_key = effective_cache_key
            if threshold_autofixed and int(thresh) != resolved_threshold:
                self.panel.set_threshold(resolved_threshold)

            self.analysis_panel.set_time_series(self._pnm_result)
            self.visualizer.set_data(self._reference_mesh)
            self.stats_panel.update_statistics(self._reference_mesh.metadata)
            self._show_tracking_summary()
            if threshold_autofixed:
                self.visualizer.update_status(
                    f"4D CT tracking complete (threshold auto-fixed to {resolved_threshold})."
                )
            else:
                self.visualizer.update_status("4D CT tracking complete.")

        def on_cancelled():
            self._close_progress_dialog()
            self._set_ui_busy(False)
            self._track_worker = None
            self.visualizer.update_status("4D CT tracking cancelled.")

        def on_error(message: str):
            self._close_progress_dialog()
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
        self.data_manager.raw_ct_data = self._volumes[index]
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
            self.visualizer.set_data(mesh, reset_camera=False, preserve_overlays=True)
            self.stats_panel.update_statistics(mesh.metadata)
            self.visualizer.update_status(f"Viewing PNM at t={index} (connectivity from t=0)")
            return

        if current_mode in ["volume", "slices", "iso"] and self.data_manager.has_segmented():
            thresh = int(self.panel.get_threshold())
            cache_key = self._pore_cache_key(index, thresh)
            if cache_key in self._pores_cache:
                pores_data = self._pores_cache[cache_key]
                self.data_manager.segmented_volume = pores_data
                self.visualizer.set_data(pores_data, reset_camera=False, preserve_overlays=True)
                self.stats_panel.update_statistics(pores_data.metadata)
                self.visualizer.update_status(f"Viewing pores at t={index} (cached)")
                return

            try:
                pores_data = self._pore_processor.process(self._volumes[index], threshold=thresh)
                self._pores_cache[cache_key] = pores_data
                self.data_manager.segmented_volume = pores_data
                self.visualizer.set_data(pores_data, reset_camera=False, preserve_overlays=True)
                self.stats_panel.update_statistics(pores_data.metadata)
                self.visualizer.update_status(f"Viewing pores at t={index}")
                return
            except Exception as exc:
                print(f"[TimeseriesHandler] Failed to extract pores for t={index}: {exc}")

        self.visualizer.set_data(self._volumes[index], reset_camera=False, preserve_overlays=True)
        self.stats_panel.update_statistics(self._volumes[index].metadata)
        self.visualizer.update_status(f"Viewing volume at t={index}")

    def _show_tracking_summary(self):
        summary = self._pnm_result.get_summary()
        eval_msg = ""
        eval_report = getattr(self._pnm_result.tracking, "evaluation", {})
        if isinstance(eval_report, dict) and eval_report.get("available"):
            overall = eval_report.get("overall", {})
            mean_voxel_iou_raw = overall.get("mean_voxel_iou")
            mean_voxel_iou_text = (
                f"{float(mean_voxel_iou_raw):.1%}"
                if isinstance(mean_voxel_iou_raw, (int, float))
                else "n/a"
            )
            mean_instance_f1 = float(
                overall.get(
                    "mean_pore_level_instance_f1",
                    overall.get("mean_instance_f1", 0.0),
                )
            )
            mean_tracking_acc = float(overall.get("mean_tracking_accuracy", 0.0))
            ref_cov = float(overall.get("t0_reference_gt_coverage", 0.0))
            novel_avg = float(overall.get("mean_untracked_novel_segments", 0.0))
            eval_msg = (
                f"\n\nSimulation-annotation pore-level evaluation:\n"
                f"- Mean voxel IoU (strict labels): {mean_voxel_iou_text}\n"
                f"- Mean pore instance F1: {mean_instance_f1:.1%}\n"
                f"- Mean tracking accuracy: {mean_tracking_acc:.1%}\n"
                f"- t0 reference->GT coverage: {ref_cov:.1%}\n"
                f"- Untracked novel segments/step: {novel_avg:.2f}\n"
                f"- Policy: fixed reference set (novel segments are diagnostics only)"
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
        if self._transform_worker is not None and self._transform_worker.isRunning():
            self._transform_worker.cancel()
            self._transform_worker.wait(2000)
        self._cancel_overlay_seg_workers()

        self._volumes.clear()
        self._pnm_result = None
        self._reference_mesh = None
        self._reference_snapshot = None
        self._current_cache_key = None
        self._pores_cache.clear()
        self._current_timepoint = 0

        if hasattr(self.visualizer, "clear_pnm_color_cache"):
            self.visualizer.clear_pnm_color_cache()

