import sys
import os
from typing import Optional, List

from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt

# Import Core Logic
from core import BaseProcessor
from processors import PoreExtractionProcessor, PoreToSphereProcessor
from data import ScientificDataManager
from exporters import VTKExporter

# Import View and UI Components
from gui import MainWindow, StructureProcessingPanel, StatisticsPanel
from gui.panels import TimeSeriesPanel

# Import Handlers
from gui.handlers.timeseries_handler import TimeseriesHandler
from gui.handlers.workflow_handler import WorkflowHandler


class AppController:
    """
    Application Controller (The 'C' in MVC).
    Orchestrates interaction between Data, Logic, and UI.
    Delegates specific business logic to Handler classes.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)

        # 1. Initialize View
        self.visualizer = MainWindow()

        # 2. Initialize Logic
        self.data_manager = ScientificDataManager()
        
        # Connect Visualizer to DataManager for centralized data flow
        self.visualizer.set_data_manager(self.data_manager)

        # 3. Setup Processing UI
        self._setup_workflow_ui()
        
        # 4. Initialize Handlers
        self.workflow_handler = WorkflowHandler(self)
        self.timeseries_handler = TimeseriesHandler(self)
        
        # 5. Connect Handler Signals
        self._connect_signals()

    def _setup_workflow_ui(self):
        self.panel = StructureProcessingPanel()

        # Add to left side (controls)
        self.visualizer.add_custom_panel(self.panel, index=1, side='left')
        
        # Add Statistics Panel to right side (info)
        self.stats_panel = StatisticsPanel()
        self.visualizer.add_custom_panel(self.stats_panel, index=1, side='right')
        
        # Add Time Series Panel to right side
        self.timeseries_panel = TimeSeriesPanel()
        self.visualizer.add_custom_panel(self.timeseries_panel, index=2, side='right')

        # GPU Control
        from core.gpu_backend import get_gpu_backend
        self.panel.gpu_toggled.connect(lambda checked: get_gpu_backend().set_enabled(checked))
        
        self.panel.export_clicked.connect(self._export_vtk_dialog)
        self.panel.reset_clicked.connect(self._reset_to_original)

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        # Workflow Signals
        self.panel.load_clicked.connect(self.workflow_handler.load_dicom_dialog)
        self.panel.fast_load_clicked.connect(self.workflow_handler.fast_load_dicom_dialog)
        self.panel.dummy_clicked.connect(self.workflow_handler.load_dummy_data)
        
        self.panel.extract_pores_clicked.connect(self._process_pores)
        self.panel.pnm_clicked.connect(self._process_spheres)
        self.panel.auto_threshold_clicked.connect(self.workflow_handler.auto_detect_threshold)
        
        # 4D CT Signals
        self.panel.load_4dct_clicked.connect(self.timeseries_handler.load_series)
        self.panel.track_4dct_clicked.connect(self.timeseries_handler.track_pores)
        self.timeseries_panel.timepoint_changed.connect(self.timeseries_handler.set_timepoint)

    # ==========================================
    # Helper Components (Shared by Handlers)
    # ==========================================

    def _create_progress_dialog(self, title: str) -> QProgressDialog:
        """Create a standard progress dialog."""
        progress = QProgressDialog(title, "Cancel", 0, 100, self.visualizer)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        return progress

    def _make_progress_callback(self, progress: QProgressDialog):
        """Create a progress callback that updates dialog and checks cancellation."""
        def callback(percent, message):
            progress.setValue(percent)
            progress.setLabelText(message)
            self.app.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("Operation cancelled by user.")
        return callback
    
    def _show_msg(self, title: str, msg: str):
        """Show information message box."""
        QMessageBox.information(self.visualizer, title, msg)

    def _show_err(self, title: str, e: Exception):
        """Show error message box."""
        QMessageBox.critical(self.visualizer, title, f"{title}: {str(e)}")
        print(f"Error: {e}")

    # ==========================================
    # Processing Logic (Delegated to Handlers)
    # ==========================================

    def _process_pores(self):
        """Extract pores from current data."""
        processor = PoreExtractionProcessor()
        
        def on_complete(segmented):
            self.data_manager.set_segmented_data(segmented)
            self.visualizer.set_data(segmented)
            meta = segmented.metadata
            msg = (
                f"Quantitative Analysis Results:\n\n"
                f"• Porosity: {meta.get('Porosity', 'N/A')}\n"
                f"• Pore Count: {meta.get('PoreCount', 0)}\n"
            )
            self._show_msg("Extraction Complete", msg)

        self.workflow_handler.run_processor_async(processor, on_complete)

    def _process_spheres(self):
        """Generate Pore Network Model from current data."""
        processor = PoreToSphereProcessor()
        
        def on_complete(pnm_data):
            self.data_manager.set_pnm_data(pnm_data)
            self.visualizer.set_data(pnm_data)
            self.stats_panel.update_statistics(pnm_data.metadata)
            
            counts = pnm_data.metadata
            msg = (
                f"Model Generated Successfully\n\n"
                f"• Nodes (Pores): {counts.get('PoreCount')}\n"
                f"• Throats: {counts.get('ConnectionCount')}\n"
                f"• Largest Pore: {counts.get('LargestPoreRatio', 'N/A')}\n"
            )
            self._show_msg("Model Generated", msg)

        self.workflow_handler.run_processor_async(processor, on_complete)

    def _reset_to_original(self):
        """Reset to original raw data."""
        if not self.data_manager.has_raw():
            return
        self.visualizer.set_data(self.data_manager.raw_ct_data)
        self.visualizer.update_status("Reset to raw data.")

    def _export_vtk_dialog(self):
        """Export current data to VTK format."""
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
            self.visualizer, "Export to VTK", default_name, file_filter
        )

        if not path:
            return

        try:
            self.visualizer.update_status(f"Exporting to {path}...")
            self.app.processEvents()
            VTKExporter.export(data_to_save, path)
            self.visualizer.update_status("Export successful.")
            self._show_msg("Export Successful", f"File saved to:\n{path}")
        except Exception as e:
            self._show_err("Export Failed", e)
            self.visualizer.update_status("Export failed.")

    def run(self, source_path: Optional[str] = None):
        """Run the application."""
        if source_path and os.path.exists(source_path):
            self.workflow_handler._load_data(source_path)
        self.visualizer.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = AppController()
    app.run()