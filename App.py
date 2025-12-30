import sys
import os
from typing import Optional
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# Import Core Logic
from Core import BaseLoader, VolumeData
from Loaders import DicomSeriesLoader, DummyLoader, FastDicomLoader
from Processors import PoreExtractionProcessor, PoreToSphereProcessor
from DataManager import ScientificDataManager
from Exporters import VTKExporter  # Import the new Exporter

# Import View and UI Components
from Visualizers import GuiVisualizer
from GUI import StructureProcessingPanel


class AppController:
    """
    Application Controller (The 'C' in MVC).
    """

    def __init__(self):
        self.app = QApplication(sys.argv)

        # 1. Initialize View
        self.visualizer = GuiVisualizer()

        # 2. Initialize Logic
        self.data_manager = ScientificDataManager()

        self.loader: Optional[BaseLoader] = None
        self.processor = PoreExtractionProcessor()
        self.sphere_processor = PoreToSphereProcessor()

        # 3. Setup Processing UI
        self._setup_workflow_ui()

    def _setup_workflow_ui(self):
        self.panel = StructureProcessingPanel()

        self.panel.load_clicked.connect(self._load_dicom_dialog)
        self.panel.fast_load_clicked.connect(self._fast_load_dicom_dialog)
        self.panel.dummy_clicked.connect(self._load_dummy_data)

        self.panel.extract_pores_clicked.connect(self._process_pores)
        self.panel.pnm_clicked.connect(self._process_spheres)
        self.panel.reset_clicked.connect(self._reset_to_original)
        self.panel.export_clicked.connect(self._export_vtk_dialog)

        self.visualizer.add_custom_panel(self.panel, index=2)

    # ==========================================
    # Logic Implementation (Slots)
    # ==========================================

    def _load_dicom_dialog(self):
        folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder")
        if folder: self._load_data(folder, fast=False)

    def _fast_load_dicom_dialog(self):
        folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder (Fast)")
        if folder: self._load_data(folder, fast=True)

    def _load_dummy_data(self):
        try:
            self.visualizer.update_status("Generating synthetic sample...")
            self.loader = DummyLoader()
            data = self.loader.load(128)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self._show_msg("Sample Loaded", "Synthetic sample ready.")
        except Exception as e:
            self._show_err("Loading Error", e)

    def _load_data(self, folder_path, fast=False):
        try:
            self.visualizer.update_status("Loading scan data...")
            self.loader = FastDicomLoader(step=2) if fast else DicomSeriesLoader()

            data = self.loader.load(folder_path)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self._show_msg("Scan Loaded",
                           f"Loaded successfully.\nMode: {'Fast' if fast else 'Standard'}")
        except Exception as e:
            self._show_err("Loading Error", e)

    def _process_pores(self):
        if not self.data_manager.has_raw():
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        try:
            self.visualizer.update_status("Extracting void space...")
            raw = self.data_manager.raw_ct_data

            is_synthetic = raw.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            segmented = self.processor.process(raw, threshold=thresh)

            self.data_manager.set_segmented_data(segmented)
            self.visualizer.set_data(segmented)

            meta = segmented.metadata
            msg = (
                f"Quantitative Analysis Results:\n\n"
                f"• Porosity: {meta.get('Porosity', 'N/A')}\n"
                f"• Pore Count: {meta.get('PoreCount', 0)}\n"
            )
            self._show_msg("Extraction Complete", msg)
        except Exception as e:
            self._show_err("Processing Error", e)

    def _process_spheres(self):
        if not self.data_manager.has_raw():
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        try:
            self.visualizer.update_status("Generating Pore Network Model (Mesh)...")
            raw = self.data_manager.raw_ct_data

            is_synthetic = raw.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            pnm_data = self.sphere_processor.process(raw, threshold=thresh)

            self.data_manager.set_pnm_data(pnm_data)
            self.visualizer.set_data(pnm_data)

            counts = pnm_data.metadata
            msg = (
                f"Model Generated Successfully (Optimized Mesh)\n\n"
                f"• Nodes (Pores): {counts.get('PoreCount')}\n"
                f"• Throats (Connections): {counts.get('ConnectionCount')}\n"
            )
            self._show_msg("Model Generated", msg)
        except Exception as e:
            self._show_err("PNM Error", e)

    def _reset_to_original(self):
        if not self.data_manager.has_raw(): return
        self.visualizer.set_data(self.data_manager.raw_ct_data)
        self.visualizer.update_status("Reset to raw data.")

    def _export_vtk_dialog(self):
        """
        Delegates the export task to VTKExporter.
        """
        data_to_save = self.visualizer.data
        if not data_to_save:
            QMessageBox.warning(self.visualizer, "Export Error", "No data loaded to export.")
            return

        # 1. Determine default filename and filter
        default_name = "output.vtk"
        file_filter = "VTK Files (*.vtk *.vtp *.vti)"

        if data_to_save.has_mesh:
            default_name = "pnm_model.vtp"  # PolyData preferred format
        elif data_to_save.raw_data is not None:
            default_name = "volume_data.vti"  # ImageData preferred format

        # 2. Open Dialog
        path, _ = QFileDialog.getSaveFileName(
            self.visualizer, "Export to VTK", default_name, file_filter
        )

        if not path:
            return

        # 3. Call Exporter
        try:
            self.visualizer.update_status(f"Exporting to {path}...")
            VTKExporter.export(data_to_save, path)

            self.visualizer.update_status("Export successful.")
            self._show_msg("Export Successful", f"File saved to:\n{path}\n\n(Includes 'IsPore' attribute if PNM)")

        except Exception as e:
            self._show_err("Export Failed", e)
            self.visualizer.update_status("Export failed.")

    def _show_msg(self, title, msg):
        QMessageBox.information(self.visualizer, title, msg)

    def _show_err(self, title, e):
        QMessageBox.critical(self.visualizer, title, f"{title}: {str(e)}")
        print(f"Error: {e}")

    def run(self, source_path: Optional[str] = None):
        if source_path and os.path.exists(source_path):
            self._load_data(source_path)
        self.visualizer.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = AppController()
    app.run()