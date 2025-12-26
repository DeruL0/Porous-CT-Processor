import os
import sys
from typing import Optional
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

# Import separated modules
from Core import BaseLoader, VolumeData
from Loaders import DicomSeriesLoader, DummyLoader, FastDicomLoader
from Processors import PoreExtractionProcessor, PoreToSphereProcessor
from Visualizers import GuiVisualizer


class AppController:
    """
    Application controller with GUI integration.
    Coordinates loader, processor, and GUI visualizer.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.visualizer = GuiVisualizer()
        self.loader: Optional[BaseLoader] = None
        self.processor = PoreExtractionProcessor()
        self.sphere_processor = PoreToSphereProcessor()

        self.original_data: Optional[VolumeData] = None
        self.current_data: Optional[VolumeData] = None

        # Add processing buttons to GUI
        self._add_processing_controls()

    def _add_processing_controls(self):
        """Add processing control buttons to the visualizer GUI"""
        # Create a processing group in the control panel
        from PyQt5.QtWidgets import QGroupBox, QPushButton, QVBoxLayout

        group = QGroupBox("Data Processing")
        layout = QVBoxLayout()

        # Load Data Button
        btn_load = QPushButton("📁 Load DICOM Series")
        btn_load.setMinimumHeight(40)
        btn_load.clicked.connect(self._load_dicom_dialog)
        layout.addWidget(btn_load)

        # Fast Load Button
        btn_fast_load = QPushButton("⚡ Fast Load (Low-Res)")
        btn_fast_load.setMinimumHeight(40)
        btn_fast_load.clicked.connect(self._fast_load_dicom_dialog)
        layout.addWidget(btn_fast_load)

        # Load Dummy Data Button
        btn_dummy = QPushButton("🧪 Load Dummy Data")
        btn_dummy.setMinimumHeight(40)
        btn_dummy.clicked.connect(self._load_dummy_data)
        layout.addWidget(btn_dummy)

        # Separator
        layout.addWidget(self.visualizer._create_separator())

        # Extract Pores Button
        btn_pores = QPushButton("🔬 Extract Pores (Raw)")
        btn_pores.setMinimumHeight(40)
        btn_pores.clicked.connect(self._process_pores)
        layout.addWidget(btn_pores)

        # Convert to Spheres Button
        btn_spheres = QPushButton("⚪ Pore Network (Spheres)")
        btn_spheres.setMinimumHeight(40)
        btn_spheres.clicked.connect(self._process_spheres)
        layout.addWidget(btn_spheres)

        # Reset to Original Button
        btn_reset = QPushButton("↩️ Reset to Original")
        btn_reset.setMinimumHeight(35)
        btn_reset.clicked.connect(self._reset_to_original)
        layout.addWidget(btn_reset)

        group.setLayout(layout)

        # Insert this group into the visualizer's control panel at position 2
        control_panel = self.visualizer.centralWidget().layout().itemAt(0).widget()
        control_panel.layout().insertWidget(2, group)

    def _load_dicom_dialog(self):
        """Open dialog to select DICOM directory and load data"""
        folder_path = QFileDialog.getExistingDirectory(
            self.visualizer,
            "Select DICOM Series Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            self._load_data(folder_path, fast=False)

    def _fast_load_dicom_dialog(self):
        """Open dialog to select DICOM directory and load with fast mode"""
        folder_path = QFileDialog.getExistingDirectory(
            self.visualizer,
            "Select DICOM Series Folder (Fast Load)",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            self._load_data(folder_path, fast=True)

    def _load_dummy_data(self):
        """Load dummy/synthetic data for testing"""
        try:
            self.visualizer.update_status("Generating synthetic data...")
            self.loader = DummyLoader()
            self.original_data = self.loader.load(128)
            self.current_data = self.original_data
            self.visualizer.set_data(self.current_data)
            QMessageBox.information(
                self.visualizer,
                "Data Loaded",
                "Synthetic dummy data loaded successfully!"
            )
        except Exception as e:
            QMessageBox.critical(
                self.visualizer,
                "Loading Error",
                f"Failed to load dummy data: {str(e)}"
            )

    def _load_data(self, folder_path: str, fast: bool = False):
        """Load DICOM data from folder"""
        try:
            self.visualizer.update_status("Loading DICOM data...")

            if fast:
                self.loader = FastDicomLoader(step=2)
                mode_text = "Fast (Low-Resolution)"
            else:
                self.loader = DicomSeriesLoader()
                mode_text = "Standard (Full-Resolution)"

            self.original_data = self.loader.load(folder_path)
            self.current_data = self.original_data
            self.visualizer.set_data(self.current_data)

            QMessageBox.information(
                self.visualizer,
                "Data Loaded",
                f"DICOM series loaded successfully!\nMode: {mode_text}\nSlices: {self.original_data.metadata.get('SliceCount', 'Unknown')}"
            )
        except Exception as e:
            QMessageBox.critical(
                self.visualizer,
                "Loading Error",
                f"Failed to load DICOM data: {str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _process_pores(self):
        """Execute pore extraction processing"""
        if self.original_data is None:
            QMessageBox.warning(
                self.visualizer,
                "No Data",
                "Please load data first before processing."
            )
            return

        try:
            self.visualizer.update_status("Extracting pores...")

            # Determine threshold based on data type
            is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            # Process
            self.current_data = self.processor.process(self.original_data, threshold=thresh)
            self.visualizer.set_data(self.current_data)

            pore_voxels = self.current_data.metadata.get('PoreVoxels', 'Unknown')
            QMessageBox.information(
                self.visualizer,
                "Processing Complete",
                f"Pore extraction complete!\nPore Voxels: {pore_voxels}\n\nUse 'Isosurface' visualization to view results."
            )
        except Exception as e:
            QMessageBox.critical(
                self.visualizer,
                "Processing Error",
                f"Failed to extract pores: {str(e)}"
            )

    def _process_spheres(self):
        """Execute pore-to-sphere network conversion"""
        if self.original_data is None:
            QMessageBox.warning(
                self.visualizer,
                "No Data",
                "Please load data first before processing."
            )
            return

        try:
            self.visualizer.update_status("Converting pores to sphere network...")

            # Determine threshold based on data type
            is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
            thresh = 500 if is_synthetic else -300

            # Process
            self.current_data = self.sphere_processor.process(self.original_data, threshold=thresh)
            self.visualizer.set_data(self.current_data)

            pore_count = self.current_data.metadata.get('PoreCount', 'Unknown')
            connection_count = self.current_data.metadata.get('ConnectionCount', 'Unknown')

            QMessageBox.information(
                self.visualizer,
                "Processing Complete",
                f"Pore network extraction complete!\n\nPores: {pore_count}\nConnections: {connection_count}\n\nUse 'Isosurface' visualization to view the sphere network."
            )
        except Exception as e:
            QMessageBox.critical(
                self.visualizer,
                "Processing Error",
                f"Failed to create pore network: {str(e)}"
            )

    def _reset_to_original(self):
        """Reset to original loaded data"""
        if self.original_data is None:
            QMessageBox.warning(
                self.visualizer,
                "No Data",
                "No original data available to reset to."
            )
            return

        self.current_data = self.original_data
        self.visualizer.set_data(self.current_data)
        self.visualizer.update_status("Reset to original data.")
        QMessageBox.information(
            self.visualizer,
            "Reset",
            "Data has been reset to original."
        )

    def run(self, source_path: Optional[str] = None):
        """Main run method - shows GUI and optionally loads initial data"""
        try:
            # If a path is provided, load it automatically
            if source_path and os.path.exists(source_path):
                print(f"Auto-loading: {source_path}")
                self._load_data(source_path, fast=False)

            # Show the GUI
            self.visualizer.show()

            # Start the Qt event loop
            sys.exit(self.app.exec_())

        except Exception as e:
            print(f"Program error: {e}")
            import traceback
            traceback.print_exc()


# ==========================================
# Program Entry Point
# ==========================================

if __name__ == "__main__":
    # Usage example
    # You can specify a path here, or leave it None to load data via GUI
    dicom_path = None  # Set to your DICOM folder path or None
    # Example: dicom_path = r'D:\zhu shuyang_25_tart_114913'

    app = AppController()
    app.run(dicom_path)