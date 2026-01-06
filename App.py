import sys
import os
from typing import Optional, Callable

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# Import Core Logic
from core import BaseLoader, VolumeData, BaseProcessor
from loaders import DicomSeriesLoader, DummyLoader, FastDicomLoader
from processors import PoreExtractionProcessor, PoreToSphereProcessor
from data import ScientificDataManager
from exporters import VTKExporter

# Import View and UI Components
from gui import MainWindow, StructureProcessingPanel, StatisticsPanel


class ProcessorWorker(QThread):
    """Background worker for running processors without blocking UI."""
    finished = pyqtSignal(object)  # Emits VolumeData result
    error = pyqtSignal(str)        # Emits error message
    progress = pyqtSignal(int, str)  # Emits (percent, message)
    
    def __init__(self, processor: BaseProcessor, data: VolumeData, threshold: int):
        super().__init__()
        self.processor = processor
        self.data = data
        self.threshold = threshold
        self._cancelled = False
    
    def run(self):
        try:
            def progress_callback(percent, message):
                if not self._cancelled:
                    self.progress.emit(percent, message)
            
            result = self.processor.process(
                self.data, 
                callback=progress_callback, 
                threshold=self.threshold
            )
            
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def cancel(self):
        self._cancelled = True



class AppController:
    """
    Application Controller (The 'C' in MVC).
    Orchestrates interaction between Data, Logic, and UI.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)

        # 1. Initialize View
        self.visualizer = MainWindow()

        # 2. Initialize Logic
        self.data_manager = ScientificDataManager()

        self.loader: Optional[BaseLoader] = None
        self.processor: BaseProcessor = PoreExtractionProcessor()
        self.sphere_processor: BaseProcessor = PoreToSphereProcessor()
        
        # Connect Visualizer to DataManager for centralized data flow
        self.visualizer.set_data_manager(self.data_manager)

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
        self.panel.auto_threshold_clicked.connect(self._auto_detect_threshold)

        # Add to left side (controls)
        self.visualizer.add_custom_panel(self.panel, index=1, side='left')
        
        # Add Statistics Panel to right side (info)
        self.stats_panel = StatisticsPanel()
        self.visualizer.add_custom_panel(self.stats_panel, index=1, side='right')
        
    def _auto_detect_threshold(self):
        """Calculate and set optimal threshold using Otsu's method."""
        current_data = self.data_manager.active_data
        if current_data is None or current_data.raw_data is None:
             QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
             return
            
        self.visualizer.update_status("Calculating optimal threshold (Otsu)...")
        self.app.processEvents()
        
        try:
            # Use static method from processor
            suggested = PoreExtractionProcessor.suggest_threshold(current_data)
            self.panel.set_threshold(suggested)
            
            self.visualizer.update_status(f"Threshold set to {suggested} HU (Otsu)")
            self._show_msg("Auto Threshold", f"Optimal threshold calculated: {suggested} HU\n\nMethod: Otsu's Binarization")
        except Exception as e:
            self._show_err("Auto Threshold Failed", e)

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
        # Create Progress Dialog
        progress = QProgressDialog("Generating synthetic sample...", "Cancel", 0, 100, self.visualizer)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def progress_callback(percent, message):
            progress.setValue(percent)
            progress.setLabelText(message)
            self.app.processEvents()
            # Check cancellation (simple check)
            if progress.wasCanceled():
                raise InterruptedError("Loading cancelled by user.")

        try:
            self.visualizer.update_status("Generating synthetic sample...")
            
            self.loader = DummyLoader()
            # Pass callback to loader
            data = self.loader.load(128, callback=progress_callback)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            
            # Set default threshold for synthetic data
            self.panel.set_threshold(500)
            
            self._show_msg("Sample Loaded", "Synthetic sample ready.")
            self.visualizer.update_status("Ready.")
        except InterruptedError:
            self.visualizer.update_status("Loading cancelled.")
        except Exception as e:
            self._show_err("Loading Error", e)
        finally:
            progress.close()

    def _load_data(self, folder_path, fast=False):
        # Create Progress Dialog
        progress = QProgressDialog("Loading scan data...", "Cancel", 0, 100, self.visualizer)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def progress_callback(percent, message):
            progress.setValue(percent)
            progress.setLabelText(message)
            self.app.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("Loading cancelled by user.")

        try:
            self.visualizer.update_status("Loading scan data...")
            
            self.loader = FastDicomLoader(step=2) if fast else DicomSeriesLoader()
            # Pass callback to loader
            data = self.loader.load(folder_path, callback=progress_callback)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            
            # Set default threshold for CT data (Air)
            self.panel.set_threshold(-300)
            
            self._show_msg("Scan Loaded",
                           f"Loaded successfully.\nMode: {'Fast' if fast else 'Standard'}")
            self.visualizer.update_status("Ready.")
        except InterruptedError:
            self.visualizer.update_status("Loading cancelled.")
        except Exception as e:
            self._show_err("Loading Error", e)
        finally:
            progress.close()

    def _run_processor_async(self, processor, data, threshold, success_callback):
        """Run processor in background thread with progress dialog."""
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self.visualizer)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        # Create and configure worker
        self.worker = ProcessorWorker(processor, data, threshold)
        
        def on_progress(percent, message):
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
            self.visualizer.update_status(message)
        
        def on_finished(result):
            self.progress_dialog.close()
            self.visualizer.update_status("Ready.")
            success_callback(result)
        
        def on_error(msg):
            self.progress_dialog.close()
            self._show_err("Processing Failed", Exception(msg))
        
        def on_cancel():
            self.worker.cancel()
            self.worker.wait()
            self.visualizer.update_status("Processing cancelled.")
        
        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.progress_dialog.canceled.connect(on_cancel)
        
        # Start background processing
        self.worker.start()

    def _process_pores(self):
        # Use DataManager's active_data (respects ROI extraction)
        current_data = self.data_manager.active_data
        if current_data is None or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        # Get threshold from UI
        thresh = self.panel.get_threshold()

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

        self._run_processor_async(self.processor, current_data, thresh, on_complete)

    def _process_spheres(self):
        # Use DataManager's active_data (respects ROI extraction)
        current_data = self.data_manager.active_data
        if current_data is None or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        # Get threshold from UI
        thresh = self.panel.get_threshold()

        def on_complete(pnm_data):
            self.data_manager.set_pnm_data(pnm_data)
            self.visualizer.set_data(pnm_data)
            
            # Update statistics panel
            self.stats_panel.update_statistics(pnm_data.metadata)
            
            counts = pnm_data.metadata
            msg = (
                f"Model Generated Successfully (Optimized Mesh)\n\n"
                f"• Nodes (Pores): {counts.get('PoreCount')}\n"
                f"• Throats (Connections): {counts.get('ConnectionCount')}\n"
                f"• Largest Pore: {counts.get('LargestPoreRatio', 'N/A')}\n"
            )
            self._show_msg("Model Generated", msg)

        self._run_processor_async(self.sphere_processor, current_data, thresh, on_complete)

    def _reset_to_original(self):
        if not self.data_manager.has_raw(): return
        self.visualizer.set_data(self.data_manager.raw_ct_data)
        self.visualizer.update_status("Reset to raw data.")

    def _export_vtk_dialog(self):
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

        if not path: return

        try:
            self.visualizer.update_status(f"Exporting to {path}...")
            # For exporting, we can leave it on main thread as usually fast enough or acceptable to block
            self.app.processEvents()
            VTKExporter.export(data_to_save, path)
            self.visualizer.update_status("Export successful.")
            self._show_msg("Export Successful", f"File saved to:\n{path}")
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