from typing import Optional
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal

from loaders import DummyLoader, SmartDicomLoader, LoadStrategy
from core import BaseProcessor
from processors import PoreExtractionProcessor

class ProcessorWorker(QThread):
    """Background worker for running processors without blocking UI."""
    finished = pyqtSignal(object)  # Emits VolumeData result
    error = pyqtSignal(str)        # Emits error message
    progress = pyqtSignal(int, str)  # Emits (percent, message)
    
    def __init__(self, processor: BaseProcessor, data, threshold: int):
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


class WorkflowHandler(QObject):
    """
    Handler for Standard Workflow operations (Load, Process, Analyze).
    """
    
    def __init__(self, main_controller):
        super().__init__()
        self.controller = main_controller
        self.visualizer = main_controller.visualizer
        self.panel = main_controller.panel
        self.data_manager = main_controller.data_manager
        
        self.worker = None
        self.progress_dialog = None


    def load_dicom_dialog(self):
        """Load DICOM with full resolution."""
        if self.panel.load_4d_check.isChecked():
            self.controller.timeseries_handler.load_series(strategy=LoadStrategy.FULL)
        else:
            folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder")
            if folder:
                self._load_data(folder, strategy=None)

    def fast_load_dicom_dialog(self):
        """Load DICOM with fast/preview mode."""
        if self.panel.load_4d_check.isChecked():
            self.controller.timeseries_handler.load_series(strategy=LoadStrategy.FAST)
        else:
            folder = QFileDialog.getExistingDirectory(self.visualizer, "Select Scan Series Folder (Fast)")
            if folder:
                self._load_data(folder, strategy=LoadStrategy.FAST)


    def load_dummy_data(self):
        """Generate and load synthetic sample data."""
        progress = self.controller._create_progress_dialog("Generating synthetic sample...")
        callback = self.controller._make_progress_callback(progress)

        try:
            self.visualizer.update_status("Generating synthetic sample...")
            
            loader = DummyLoader()
            data = loader.load(128, callback=callback)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self.panel.set_threshold(500)
            
            self.controller._show_msg("Sample Loaded", "Synthetic sample ready.")
            self.visualizer.update_status("Ready.")
        except Exception as e:
            self.controller._show_err("Loading Error", e)
        finally:
            progress.close()

    def _load_data(self, folder_path: str, strategy: Optional[LoadStrategy] = None):
        """Executes data loading."""
        progress = self.controller._create_progress_dialog("Loading scan data...")
        callback = self.controller._make_progress_callback(progress)

        try:
            self.visualizer.update_status("Loading scan data...")
            
            loader = SmartDicomLoader(strategy=strategy)
            data = loader.load(folder_path, callback=callback)

            self.data_manager.load_raw_data(data)
            self.visualizer.set_data(data)
            self.panel.set_threshold(-300)
            
            mode = data.metadata.get('LoadStrategy', 'Auto')
            self.controller._show_msg("Scan Loaded", f"Loaded successfully.\nMode: {mode}")
            self.visualizer.update_status("Ready.")
        except Exception as e:
            self.controller._show_err("Loading Error", e)
        finally:
            progress.close()

    def auto_detect_threshold(self):
        """Run auto-threshold detection."""
        current_data = self.data_manager.active_data
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return
        
        algorithm = self.panel.get_algorithm()
        self.visualizer.update_status(f"Calculating threshold ({algorithm})...")
        self.controller.app.processEvents()
        
        try:
            suggested = PoreExtractionProcessor.suggest_threshold(current_data, algorithm)
            self.panel.set_threshold(suggested)
            
            algo_display = algorithm.capitalize() if algorithm != 'auto' else 'Auto'
            self.visualizer.update_status(f"Threshold set to {suggested} HU ({algo_display})")
            self.controller._show_msg("Threshold Results", f"Threshold: {suggested} HU\nAlgorithm: {algo_display}")
        except Exception as e:
            self.controller._show_err("Threshold Detection Failed", e)

    def run_processor_async(self, processor, success_callback):
        """Run a processor in background."""
        current_data = self.data_manager.active_data
        if not current_data or current_data.raw_data is None:
            QMessageBox.warning(self.visualizer, "No Data", "Please load a sample first.")
            return

        thresh = self.panel.get_threshold()
        
        self.progress_dialog = self.controller._create_progress_dialog("Processing...")
        
        # Create and configure worker
        self.worker = ProcessorWorker(processor, current_data, thresh)
        
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
            self.controller._show_err("Processing Failed", Exception(msg))
        
        def on_cancel():
            self.worker.cancel()
            self.worker.wait()
            self.visualizer.update_status("Processing cancelled.")
        
        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.progress_dialog.canceled.connect(on_cancel)
        
        self.worker.start()
