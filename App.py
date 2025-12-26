import sys
import os
from typing import Optional
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# Import Core Logic
from Core import BaseLoader, VolumeData
from Loaders import DicomSeriesLoader, DummyLoader, FastDicomLoader
from Processors import PoreExtractionProcessor, PoreToSphereProcessor

# Import View and UI Components
from Visualizers import GuiVisualizer
from GUI import StructureProcessingPanel


class AppController:
    """
    Application Controller (The 'C' in MVC).
    Coordinates:
    1. Processing Logic (Loaders, Processors)
    2. The View (GuiVisualizer)
    3. User Actions (StructureProcessingPanel)
    """

    def __init__(self):
        self.app = QApplication(sys.argv)

        # 1. Initialize View
        self.visualizer = GuiVisualizer()

        # 2. Initialize Logic
        self.loader: Optional[BaseLoader] = None
        self.processor = PoreExtractionProcessor()
        self.sphere_processor = PoreToSphereProcessor()

        self.original_data: Optional[VolumeData] = None
        self.current_data: Optional[VolumeData] = None

        # 3. Setup Processing UI and Connect Signals
        self._setup_workflow_ui()

    def _setup_workflow_ui(self):
        """Create the processing panel and connect it to controller methods"""
        self.panel = StructureProcessingPanel()

        # Connect Signals from UI -> Controller Methods
        self.panel.load_clicked.connect(self._load_dicom_dialog)
        self.panel.fast_load_clicked.connect(self._fast_load_dicom_dialog)
        self.panel.dummy_clicked.connect(self._load_dummy_data)

        self.panel.extract_pores_clicked.connect(self._process_pores)
        self.panel.pnm_clicked.connect(self._process_spheres)
        self.panel.reset_clicked.connect(self._reset_to_original)

        # Inject this panel into the Visualizer
        # We insert it at index 2 (After Title and Separator, Before Info)
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
            self.original_data = self.loader.load(128)
            self._update_view_data(self.original_data)
            self._show_msg("Sample Loaded", "Synthetic sample ready.")
        except Exception as e:
            self._show_err("Loading Error", e)

    def _load_data(self, folder_path, fast=False):
        try:
            self.visualizer.update_status("Loading scan data...")
            self.loader = FastDicomLoader(step=2) if fast else DicomSeriesLoader()

            self.original_data = self.loader.load(folder_path)
            self._update_view_data(self.original_data)

            self._show_msg("Scan Loaded",
                           f"Loaded successfully.\nMode: {'Fast' if fast else 'Standard'}")
        except Exception as e:
            self._show_err("Loading Error", e)

    def _process_pores(self):
        if not self._check_data(): return
        try:
            self.visualizer.update_status("Extracting void space...")

            # Auto-detect threshold context
            is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            self.current_data = self.processor.process(self.original_data, threshold=thresh)
            self.visualizer.set_data(self.current_data)

            voxels = self.current_data.metadata.get('PoreVoxels', 0)
            self._show_msg("Extraction Complete", f"Void voxels: {voxels}\nVisualize using Isosurface.")
        except Exception as e:
            self._show_err("Processing Error", e)

    def _process_spheres(self):
        if not self._check_data(): return
        try:
            self.visualizer.update_status("Generating Pore Network Model...")

            is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            self.current_data = self.sphere_processor.process(self.original_data, threshold=thresh)
            self.visualizer.set_data(self.current_data)

            counts = self.current_data.metadata
            self._show_msg("Model Generated",
                           f"Nodes: {counts.get('PoreCount')}\nThroats: {counts.get('ConnectionCount')}")
        except Exception as e:
            self._show_err("PNM Error", e)

    def _reset_to_original(self):
        if not self._check_data(): return
        self._update_view_data(self.original_data)
        self.visualizer.update_status("Reset to raw data.")

    # Helpers
    def _update_view_data(self, data):
        self.current_data = data
        self.visualizer.set_data(self.current_data)

    def _check_data(self):
        if self.original_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return False
        return True

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