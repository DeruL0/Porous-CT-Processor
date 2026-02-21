import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from data import ScientificDataManager
from exporters import VTKExporter
from gui import MainWindow, StatisticsPanel, StructureProcessingPanel
from gui.panels import TimeSeriesControlPanel, TrackingAnalysisPanel
from gui.handlers.timeseries_handler import TimeseriesHandler
from gui.handlers.workflow_handler import WorkflowHandler


class AppController:
    """
    Application controller (MVC C layer).
    Coordinates UI state and delegates workflow logic to handlers.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)

        self.visualizer = MainWindow()
        self.data_manager = ScientificDataManager()
        self.visualizer.set_data_manager(self.data_manager)

        self._setup_workflow_ui()

        self.workflow_handler = WorkflowHandler(self)
        self.timeseries_handler = TimeseriesHandler(self)
        self._connect_signals()

    def _setup_workflow_ui(self):
        self.panel = StructureProcessingPanel()
        self.visualizer.add_custom_panel(self.panel, index=1, side="left")

        self.timeseries_control = TimeSeriesControlPanel()
        self.visualizer.add_custom_panel(self.timeseries_control, index=2, side="left")

        self.stats_panel = StatisticsPanel()
        self.visualizer.add_custom_panel(self.stats_panel, index=1, side="right")

        self.tracking_analysis = TrackingAnalysisPanel()
        self.visualizer.add_custom_panel(self.tracking_analysis, index=2, side="right")

        from core.gpu_backend import get_gpu_backend

        self.panel.gpu_toggled.connect(lambda checked: get_gpu_backend().set_enabled(checked))
        self.panel.export_clicked.connect(self._export_vtk_dialog)
        self.panel.reset_clicked.connect(self._reset_to_original)

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        self.panel.load_clicked.connect(self.workflow_handler.load_dicom_dialog)
        self.panel.fast_load_clicked.connect(self.workflow_handler.fast_load_dicom_dialog)
        self.panel.dummy_clicked.connect(self.workflow_handler.load_dummy_data)
        self.panel.extract_pores_clicked.connect(self._process_pores)
        self.panel.pnm_clicked.connect(self._process_spheres)
        self.panel.auto_threshold_clicked.connect(self.workflow_handler.auto_detect_threshold)

        self.timeseries_control.timepoint_changed.connect(self.timeseries_handler.set_timepoint)

    def _create_progress_dialog(self, title: str) -> QProgressDialog:
        """Create a standard progress dialog."""
        progress = QProgressDialog(title, "Cancel", 0, 100, self.visualizer)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        return progress

    def _show_msg(self, title: str, msg: str):
        """Show information message box."""
        QMessageBox.information(self.visualizer, title, msg)

    def _show_err(self, title: str, exc: Exception):
        """Show error message box."""
        QMessageBox.critical(self.visualizer, title, f"{title}: {str(exc)}")
        print(f"Error: {exc}")

    def _process_pores(self):
        """Extract pores from active volume using shared DAG pipeline."""

        def on_complete(results):
            segmented = results.get("segment")
            if segmented is None:
                raise ValueError("Segmentation result not found in pipeline output.")

            self.data_manager.set_segmented_data(segmented)
            self.visualizer.set_data(segmented)

            meta = segmented.metadata
            msg = (
                "Quantitative Analysis Results:\n\n"
                f"Porosity: {meta.get('Porosity', 'N/A')}\n"
                f"Pore Count: {meta.get('PoreCount', 0)}\n"
            )
            self._show_msg("Extraction Complete", msg)

        self.workflow_handler.run_pipeline_async(
            target_stage="segment",
            success_callback=on_complete,
            title="Extracting pores...",
        )

    def _process_spheres(self):
        """Generate pore network model from active data or run 4D tracking."""
        if self.timeseries_handler.has_volumes:
            self.timeseries_handler.track_pores()
            return

        def on_complete(results):
            segmented = results.get("segment")
            pnm_data = results.get("pnm")
            if pnm_data is None:
                raise ValueError("PNM result not found in pipeline output.")

            if segmented is not None:
                self.data_manager.set_segmented_data(segmented)
            self.data_manager.set_pnm_data(pnm_data)
            self.visualizer.set_data(pnm_data)
            self.stats_panel.update_statistics(pnm_data.metadata)

            counts = pnm_data.metadata
            msg = (
                "Model Generated Successfully\n\n"
                f"Nodes (Pores): {counts.get('PoreCount')}\n"
                f"Throats: {counts.get('ConnectionCount')}\n"
                f"Largest Pore: {counts.get('LargestPoreRatio', 'N/A')}\n"
            )
            self._show_msg("Model Generated", msg)

        self.workflow_handler.run_pipeline_async(
            target_stage="pnm",
            success_callback=on_complete,
            title="Generating PNM model...",
        )

    def _reset_to_original(self):
        """Reset view to original raw volume."""
        if not self.data_manager.has_raw():
            return
        self.visualizer.set_data(self.data_manager.raw_ct_data)
        self.visualizer.update_status("Reset to raw data.")

    def _export_vtk_dialog(self):
        """Export current visualized data to VTK format."""
        data_to_save = self.visualizer.data
        if not data_to_save:
            QMessageBox.warning(self.visualizer, "Export Error", "No data loaded to export.")
            return

        default_name = "output.vtk"
        file_filter = "VTK Files (*.vtk *.vtp *.vti)"
        if data_to_save.has_mesh:
            default_name = "pnm_model.vtp"
        elif data_to_save.raw_data is not None:
            default_name = "volume_data.vti"

        path, _ = QFileDialog.getSaveFileName(
            self.visualizer,
            "Export to VTK",
            default_name,
            file_filter,
        )
        if not path:
            return

        try:
            self.visualizer.update_status(f"Exporting to {path}...")
            VTKExporter.export(data_to_save, path)
            self.visualizer.update_status("Export successful.")
            self._show_msg("Export Successful", f"File saved to:\n{path}")
        except Exception as exc:
            self._show_err("Export Failed", exc)
            self.visualizer.update_status("Export failed.")

    def run(self, source_path: Optional[str] = None):
        """Run the application."""
        if source_path and os.path.exists(source_path):
            self.workflow_handler.load_dicom_path(source_path)
        self.visualizer.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = AppController()
    app.run()
