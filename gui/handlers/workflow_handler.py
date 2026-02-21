from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core import VolumeProcessDTO, resolve_pipeline_stages, run_volume_pipeline
from core.progress import CancelFlagObserver, ProgressBus, StageProgressMapper
from gui.progress_observers import QtSignalProgressObserver
from loaders import DummyLoader, LoadStrategy, SmartDicomLoader
from processors import PoreExtractionProcessor


class PipelineWorker(QThread):
    """Background worker for shared DAG pipeline execution."""

    finished = pyqtSignal(object)  # dict[str, Any]
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, dto: VolumeProcessDTO, input_data, target_stage: str):
        super().__init__()
        self.dto = dto
        self.input_data = input_data
        self.target_stage = target_stage
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            stages = resolve_pipeline_stages(target_stage=self.target_stage, include_export=False)
            mapper = StageProgressMapper(stages)
            progress_bus = ProgressBus()
            progress_bus.subscribe(QtSignalProgressObserver(self.progress.emit, mapper=mapper, stage_prefix=True))
            progress_bus.subscribe(CancelFlagObserver(lambda: self._cancelled))

            results = run_volume_pipeline(
                dto=self.dto,
                input_data=self.input_data,
                target_stage=self.target_stage,
                include_export=False,
                progress_bus=progress_bus,
            )

            if self._cancelled:
                self.cancelled.emit()
                return
            self.progress.emit(100, "Ready.")
            self.finished.emit(results)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class VolumeLoadWorker(QThread):
    """Background worker for synthetic/DICOM loading."""

    finished = pyqtSignal(object)  # VolumeData
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, mode: str, folder_path: Optional[str] = None, strategy: Optional[LoadStrategy] = None):
        super().__init__()
        self.mode = mode
        self.folder_path = folder_path
        self.strategy = strategy
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            def callback(percent: int, message: str) -> None:
                if self._cancelled:
                    raise InterruptedError("Operation cancelled by user.")
                self.progress.emit(max(0, min(100, int(percent))), message)

            if self.mode == "dummy":
                data = DummyLoader().load(128, callback=callback)
            elif self.mode == "dicom":
                if not self.folder_path:
                    raise ValueError("folder_path is required for dicom loading")
                data = SmartDicomLoader(strategy=self.strategy).load(self.folder_path, callback=callback)
            else:
                raise ValueError(f"Unknown load mode: {self.mode}")

            if self._cancelled:
                self.cancelled.emit()
                return
            self.finished.emit(data)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class ThresholdDetectWorker(QThread):
    """Background worker for auto-threshold detection."""

    finished = pyqtSignal(int)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, data, algorithm: str):
        super().__init__()
        self.data = data
        self.algorithm = algorithm
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if self._cancelled:
                self.cancelled.emit()
                return
            self.progress.emit(0, f"Calculating threshold ({self.algorithm})...")
            suggested = PoreExtractionProcessor.suggest_threshold(self.data, self.algorithm)
            if self._cancelled:
                self.cancelled.emit()
                return
            self.progress.emit(100, "Threshold detection complete.")
            self.finished.emit(int(suggested))
        except Exception as exc:
            self.error.emit(str(exc))


class WorkflowHandler(QObject):
    """
    Handler for standard workflow operations (load, segment, model).
    """

    def __init__(self, main_controller):
        super().__init__()
        self.controller = main_controller
        self.visualizer = main_controller.visualizer
        self.panel = main_controller.panel
        self.data_manager = main_controller.data_manager

        self._pipeline_worker: Optional[PipelineWorker] = None
        self._load_worker: Optional[VolumeLoadWorker] = None
        self._threshold_worker: Optional[ThresholdDetectWorker] = None
        self._progress_dialog = None

    @property
    def is_busy(self) -> bool:
        workers = (self._pipeline_worker, self._load_worker, self._threshold_worker)
        return any(w is not None and w.isRunning() for w in workers)

    def _set_ui_busy(self, busy: bool) -> None:
        self.panel.setEnabled(not busy)

    def _ensure_idle(self) -> bool:
        if self.is_busy:
            QMessageBox.warning(self.visualizer, "Busy", "Another workflow task is currently running.")
            return False
        if hasattr(self.controller, "timeseries_handler") and self.controller.timeseries_handler.is_busy:
            QMessageBox.warning(self.visualizer, "Busy", "Time-series operation is currently running.")
            return False
        return True

    def load_dicom_dialog(self):
        """Load DICOM with full resolution."""
        if not self._ensure_idle():
            return

        if self.panel.load_4d_check.isChecked():
            self.controller.timeseries_handler.load_series(strategy=LoadStrategy.FULL)
            return

        folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder")
        if folder:
            self.load_dicom_path(folder, strategy=None)

    def fast_load_dicom_dialog(self):
        """Load DICOM with fast/preview mode."""
        if not self._ensure_idle():
            return

        if self.panel.load_4d_check.isChecked():
            self.controller.timeseries_handler.load_series(strategy=LoadStrategy.FAST)
            return

        folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder (Fast)")
        if folder:
            self.load_dicom_path(folder, strategy=LoadStrategy.FAST)

    def load_dicom_path(self, folder_path: str, strategy: Optional[LoadStrategy] = None) -> None:
        """Load DICOM from a known folder path asynchronously."""
        title = "Loading scan data (fast)..." if strategy == LoadStrategy.FAST else "Loading scan data..."
        self._run_load_worker(mode="dicom", title=title, folder_path=folder_path, strategy=strategy)

    # Backward compatibility for legacy caller App.run(...)
    def _load_data(self, folder_path: str, strategy: Optional[LoadStrategy] = None):
        self.load_dicom_path(folder_path, strategy=strategy)

    def load_dummy_data(self):
        """Generate and load synthetic sample data."""
        if not self._ensure_idle():
            return
        self._run_load_worker(mode="dummy", title="Generating synthetic sample...")

    def _run_load_worker(
        self,
        mode: str,
        title: str,
        folder_path: Optional[str] = None,
        strategy: Optional[LoadStrategy] = None,
    ) -> None:
        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog(title)
        self._load_worker = VolumeLoadWorker(mode=mode, folder_path=folder_path, strategy=strategy)

        def on_progress(percent: int, message: str):
            self._progress_dialog.setValue(percent)
            self._progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)

        def on_finished(data):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            if mode == "dummy":
                self.panel.set_threshold(500)
                self.controller._show_msg("Sample Loaded", "Synthetic sample ready.")
            else:
                self.panel.set_threshold(-300)
                load_mode = data.metadata.get("LoadStrategy", "Auto")
                self.controller._show_msg("Scan Loaded", f"Loaded successfully.\nMode: {load_mode}")
            self.visualizer.update_status("Ready.")

        def on_cancelled():
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None
            self.visualizer.update_status("Loading cancelled.")

        def on_error(message: str):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._load_worker = None
            self.controller._show_err("Loading Error", Exception(message))

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

    def auto_detect_threshold(self):
        """Run auto-threshold detection for active data."""
        if not self._ensure_idle():
            return

        current_data = self.data_manager.active_data
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        algorithm = self.panel.get_algorithm()
        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog("Detecting threshold...")
        self._threshold_worker = ThresholdDetectWorker(current_data, algorithm)

        def on_progress(percent: int, message: str):
            self._progress_dialog.setValue(percent)
            self._progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)

        def on_finished(suggested: int):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._threshold_worker = None

            self.panel.set_threshold(suggested)
            algo_display = algorithm.capitalize() if algorithm != "auto" else "Auto"
            self.visualizer.update_status(f"Threshold set to {suggested} HU ({algo_display})")
            self.controller._show_msg(
                "Threshold Results",
                f"Threshold: {suggested} HU\nAlgorithm: {algo_display}",
            )

        def on_cancelled():
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._threshold_worker = None
            self.visualizer.update_status("Threshold detection cancelled.")

        def on_error(message: str):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._threshold_worker = None
            self.controller._show_err("Threshold Detection Failed", Exception(message))

        def on_cancel_requested():
            if self._threshold_worker is not None:
                self._threshold_worker.cancel()
                self.visualizer.update_status("Cancelling...")

        self._threshold_worker.progress.connect(on_progress)
        self._threshold_worker.finished.connect(on_finished)
        self._threshold_worker.cancelled.connect(on_cancelled)
        self._threshold_worker.error.connect(on_error)
        self._progress_dialog.canceled.connect(on_cancel_requested)
        self._threshold_worker.start()

    def _build_runtime_dto(self) -> VolumeProcessDTO:
        """Create processing DTO from current GUI controls."""
        return VolumeProcessDTO(
            threshold=float(self.panel.get_threshold()),
            auto_threshold=False,
            export_formats=tuple(),
        )

    def run_pipeline_async(
        self,
        target_stage: str,
        success_callback: Callable[[dict], None],
        title: str = "Processing...",
    ) -> None:
        """Run shared DAG pipeline asynchronously on current active data."""
        if not self._ensure_idle():
            return

        current_data = self.data_manager.active_data
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog(title)
        self._pipeline_worker = PipelineWorker(
            dto=self._build_runtime_dto(),
            input_data=current_data,
            target_stage=target_stage,
        )

        def on_progress(percent: int, message: str):
            self._progress_dialog.setValue(percent)
            self._progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)

        def on_finished(results: dict):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._pipeline_worker = None
            self.visualizer.update_status("Ready.")
            success_callback(results)

        def on_cancelled():
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._pipeline_worker = None
            self.visualizer.update_status("Processing cancelled.")

        def on_error(message: str):
            self._progress_dialog.close()
            self._set_ui_busy(False)
            self._pipeline_worker = None
            self.controller._show_err("Processing Failed", Exception(message))

        def on_cancel_requested():
            if self._pipeline_worker is not None:
                self._pipeline_worker.cancel()
                self.visualizer.update_status("Cancelling...")

        self._pipeline_worker.progress.connect(on_progress)
        self._pipeline_worker.finished.connect(on_finished)
        self._pipeline_worker.cancelled.connect(on_cancelled)
        self._pipeline_worker.error.connect(on_error)
        self._progress_dialog.canceled.connect(on_cancel_requested)
        self._pipeline_worker.start()

    def run_processor_async(self, processor, success_callback):
        """
        Backward compatibility wrapper.

        Maps old processor-based API to shared DAG execution:
        - PoreExtractionProcessor -> stage 'segment'
        - PoreToSphereProcessor   -> stage 'pnm'
        """
        processor_name = type(processor).__name__
        if processor_name == "PoreExtractionProcessor":
            stage = "segment"
        elif processor_name == "PoreToSphereProcessor":
            stage = "pnm"
        else:
            raise ValueError(f"Unsupported processor for async workflow: {processor_name}")

        def adapt(results: dict):
            success_callback(results.get(stage))

        self.run_pipeline_async(target_stage=stage, success_callback=adapt)
