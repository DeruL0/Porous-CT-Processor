from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core import VolumeProcessDTO, resolve_pipeline_stages, run_volume_pipeline
from core.progress import CancelFlagObserver, ProgressBus, StageProgressMapper
from gui.progress_observers import QtSignalProgressObserver
from loaders import LoadStrategy


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


class WorkflowHandler(QObject):
    """
    Handler for standard workflow operations (load, threshold, segment, model).
    """

    def __init__(self, main_controller):
        super().__init__()
        self.controller = main_controller
        self.visualizer = main_controller.visualizer
        self.panel = main_controller.panel
        self.data_manager = main_controller.data_manager

        self._pipeline_worker: Optional[PipelineWorker] = None
        self._progress_dialog = None

    @property
    def is_busy(self) -> bool:
        return self._pipeline_worker is not None and self._pipeline_worker.isRunning()

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

    def _resolve_input_data(self):
        """
        Resolve runtime input data for processing.

        In 4D mode, always prefer the currently selected timepoint volume.
        """
        ts_handler = getattr(self.controller, "timeseries_handler", None)
        if ts_handler is not None and getattr(ts_handler, "has_volumes", False):
            current = ts_handler.get_current_raw_volume()
            if current is not None and current.raw_data is not None:
                return current
        return self.data_manager.active_data

    def _start_pipeline_task(
        self,
        *,
        dto: VolumeProcessDTO,
        target_stage: str,
        title: str,
        success_callback: Callable[[dict], None],
        error_title: str,
        cancel_status: str,
        ready_status: Optional[str] = "Ready.",
        input_data=None,
    ) -> None:
        self._set_ui_busy(True)
        self._progress_dialog = self.controller._create_progress_dialog(title)
        self._pipeline_worker = PipelineWorker(
            dto=dto,
            input_data=input_data,
            target_stage=target_stage,
        )

        def finalize():
            dialog = self._progress_dialog
            self._progress_dialog = None
            if dialog is not None:
                try:
                    dialog.close()
                except Exception:
                    pass
            self._set_ui_busy(False)
            self._pipeline_worker = None

        def on_progress(percent: int, message: str):
            dialog = self._progress_dialog
            if dialog is not None:
                try:
                    dialog.setValue(percent)
                    dialog.setLabelText(message)
                except (RuntimeError, AttributeError):
                    # Dialog may already be closed by finish/cancel path.
                    pass
            self.visualizer.update_status(message)

        def on_finished(results: dict):
            finalize()
            try:
                success_callback(results)
                if ready_status:
                    self.visualizer.update_status(ready_status)
            except Exception as exc:
                self.controller._show_err(error_title, exc)

        def on_cancelled():
            finalize()
            self.visualizer.update_status(cancel_status)

        def on_error(message: str):
            finalize()
            self.controller._show_err(error_title, Exception(message))

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

    @staticmethod
    def _strategy_value(strategy: Optional[LoadStrategy]) -> str:
        if strategy is None:
            return LoadStrategy.AUTO.value
        return strategy.value

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
        """Load DICOM from a known folder path asynchronously through shared DAG."""
        if not self._ensure_idle():
            return

        title = "Loading scan data (fast)..." if strategy == LoadStrategy.FAST else "Loading scan data..."
        dto = VolumeProcessDTO(
            input_path=folder_path,
            loader_type="dicom",
            load_strategy=self._strategy_value(strategy),
            export_formats=tuple(),
        )

        def on_complete(results: dict) -> None:
            data = results.get("load")
            if data is None:
                raise ValueError("Load stage did not return volume data.")

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self.panel.set_threshold(-300)
            load_mode = data.metadata.get("LoadStrategy", "Auto")
            self.controller._show_msg("Scan Loaded", f"Loaded successfully.\nMode: {load_mode}")

        self._start_pipeline_task(
            dto=dto,
            target_stage="load",
            input_data=None,
            title=title,
            success_callback=on_complete,
            error_title="Loading Error",
            cancel_status="Loading cancelled.",
        )

    # Backward compatibility for legacy caller App.run(...)
    def _load_data(self, folder_path: str, strategy: Optional[LoadStrategy] = None):
        self.load_dicom_path(folder_path, strategy=strategy)

    def load_dummy_data(self):
        """Generate and load synthetic sample data through shared DAG."""
        if not self._ensure_idle():
            return

        dto = VolumeProcessDTO(
            input_path="128",
            loader_type="dummy",
            export_formats=tuple(),
        )

        def on_complete(results: dict) -> None:
            data = results.get("load")
            if data is None:
                raise ValueError("Load stage did not return volume data.")

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self.panel.set_threshold(500)
            self.controller._show_msg("Sample Loaded", "Synthetic sample ready.")

        self._start_pipeline_task(
            dto=dto,
            target_stage="load",
            input_data=None,
            title="Generating synthetic sample...",
            success_callback=on_complete,
            error_title="Loading Error",
            cancel_status="Loading cancelled.",
        )

    def auto_detect_threshold(self):
        """Run auto-threshold detection through shared DAG threshold stage."""
        if not self._ensure_idle():
            return

        current_data = self._resolve_input_data()
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        algorithm = self.panel.get_algorithm()
        dto = VolumeProcessDTO(
            threshold=float(self.panel.get_threshold()),
            auto_threshold=True,
            threshold_algorithm=algorithm,
            export_formats=tuple(),
        )

        def on_complete(results: dict) -> None:
            suggested = int(results.get("threshold"))
            self.panel.set_threshold(suggested)
            algo_display = algorithm.capitalize() if algorithm != "auto" else "Auto"
            self.visualizer.update_status(f"Threshold set to {suggested} ({algo_display})")
            self.controller._show_msg(
                "Threshold Results",
                f"Threshold: {suggested}\nAlgorithm: {algo_display}",
            )

        self._start_pipeline_task(
            dto=dto,
            target_stage="threshold",
            input_data=current_data,
            title="Detecting threshold...",
            success_callback=on_complete,
            error_title="Threshold Detection Failed",
            cancel_status="Threshold detection cancelled.",
            ready_status=None,
        )

    def _build_runtime_dto(self) -> VolumeProcessDTO:
        """Create processing DTO from current GUI controls."""
        return VolumeProcessDTO(
            threshold=float(self.panel.get_threshold()),
            auto_threshold=False,
            threshold_algorithm=str(self.panel.get_algorithm()),
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

        current_data = self._resolve_input_data()
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        self._start_pipeline_task(
            dto=self._build_runtime_dto(),
            target_stage=target_stage,
            input_data=current_data,
            title=title,
            success_callback=success_callback,
            error_title="Processing Failed",
            cancel_status="Processing cancelled.",
        )

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

