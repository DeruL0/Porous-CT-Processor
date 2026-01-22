"""
Time series DICOM loader for 4D CT analysis.

Loads multiple DICOM folders representing different timepoints
of the same sample (e.g., compression sequence).
"""

import os
import re
from typing import List, Optional, Callable, Tuple
from pathlib import Path

from core import VolumeData, BaseLoader
from loaders.dicom import SmartDicomLoader


class TimeSeriesDicomLoader:
    """
    Load multiple DICOM folders as a time series.
    
    Expected folder structure:
    parent_folder/
    ├── t0/         (or any name, sorted by selected mode)
    │   ├── slice001.dcm
    │   └── ...
    ├── t1/
    │   ├── slice001.dcm
    │   └── ...
    └── t2/
        └── ...
    
    Supports multiple sorting modes:
    - alphabetical: Sort by folder name A-Z
    - numeric: Extract numbers from names (t1, t2, t10...)
    - date modified: Sort by file modification time
    - manual: User selects order in dialog
    """
    
    def __init__(self):
        self._loader = SmartDicomLoader()
    
    def load_series(self, 
                   parent_folder: str,
                   sort_mode: str = 'alphabetical',
                   manual_order: Optional[List[str]] = None,
                   callback: Optional[Callable[[int, str], None]] = None) -> List[VolumeData]:
        """
        Load all timepoint folders from a parent directory.
        
        Args:
            parent_folder: Path to folder containing timepoint subfolders
            sort_mode: 'alphabetical', 'numeric', 'date modified', or 'manual'
            manual_order: List of folder paths in desired order (for manual mode)
            callback: Progress callback (percent, message)
            
        Returns:
            List of VolumeData objects, one per timepoint, in time order
        """
        if not os.path.isdir(parent_folder):
            raise ValueError(f"Parent folder does not exist: {parent_folder}")
        
        # Find all subdirectories with DICOM files
        subfolders = []
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            if os.path.isdir(item_path):
                if self._contains_dicom(item_path):
                    subfolders.append(item_path)
        
        if not subfolders:
            raise ValueError(f"No DICOM subfolders found in: {parent_folder}")
        
        # Sort based on mode
        if manual_order is not None:
            # Use provided manual order
            subfolders = manual_order
        else:
            subfolders = self._sort_folders(subfolders, sort_mode)
        
        print(f"[TimeSeriesLoader] Found {len(subfolders)} timepoints, sort mode: {sort_mode}")
        for i, folder in enumerate(subfolders):
            print(f"  t={i}: {os.path.basename(folder)}")
        
        volumes = []
        total = len(subfolders)
        
        for i, folder_path in enumerate(subfolders):
            folder_name = os.path.basename(folder_path)
            
            def timepoint_callback(percent, msg):
                base_progress = int(100 * i / total)
                folder_progress = int(percent / total)
                overall = base_progress + folder_progress
                if callback:
                    callback(overall, f"[t={i}] {msg}")
                print(f"[t={i}] {msg}")
            
            if callback:
                callback(int(100 * i / total), f"Loading timepoint {i}: {folder_name}")
            
            try:
                volume = self._loader.load(folder_path, callback=timepoint_callback)
                volume.metadata['time_index'] = i
                volume.metadata['source_folder'] = folder_path
                volume.metadata['folder_name'] = folder_name
                volumes.append(volume)
                print(f"[TimeSeriesLoader] Loaded t={i}: {folder_name}, shape={volume.dimensions}")
            except Exception as e:
                print(f"[TimeSeriesLoader] Failed to load t={i} ({folder_name}): {e}")
        
        if not volumes:
            raise ValueError("Failed to load any timepoints")
        
        if callback:
            callback(100, f"Loaded {len(volumes)} timepoints")
        
        return volumes
    
    def _sort_folders(self, folders: List[str], mode: str) -> List[str]:
        """Sort folders based on the specified mode."""
        mode = mode.lower()
        
        if mode == 'numeric':
            # Extract numbers from folder names and sort numerically
            def extract_number(path):
                name = os.path.basename(path)
                # Find all numbers in the name
                numbers = re.findall(r'\d+', name)
                if numbers:
                    # Use the first number found
                    return int(numbers[0])
                return 0
            return sorted(folders, key=extract_number)
        
        elif mode == 'date modified':
            # Sort by modification time of the folder
            return sorted(folders, key=lambda p: os.path.getmtime(p))
        
        else:  # alphabetical (default)
            return sorted(folders)
    
    def _contains_dicom(self, folder_path: str) -> bool:
        """Check if folder contains DICOM files."""
        for item in os.listdir(folder_path):
            if item.lower().endswith(('.dcm', '.dicom')):
                return True
            # Also check for files without extension (common in DICOM)
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) and '.' not in item:
                try:
                    import pydicom
                    pydicom.dcmread(item_path, stop_before_pixels=True)
                    return True
                except:
                    pass
        return False
    
    def get_folder_list(self, parent_folder: str, sort_mode: str = 'alphabetical') -> List[Tuple[int, str, str]]:
        """
        Get list of timepoint folders without loading data.
        
        Returns:
            List of (time_index, folder_name, folder_path) tuples
        """
        if not os.path.isdir(parent_folder):
            return []
        
        subfolders = []
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            if os.path.isdir(item_path) and self._contains_dicom(item_path):
                subfolders.append(item_path)
        
        subfolders = self._sort_folders(subfolders, sort_mode)
        
        return [(i, os.path.basename(f), f) for i, f in enumerate(subfolders)]


def load_time_series(parent_folder: str,
                    sort_mode: str = 'alphabetical',
                    callback: Optional[Callable[[int, str], None]] = None) -> List[VolumeData]:
    """
    Convenience function to load a 4D CT time series.
    
    Args:
        parent_folder: Path to folder containing timepoint subfolders
        sort_mode: 'alphabetical', 'numeric', 'date modified', or 'manual'
        callback: Progress callback (percent, message)
        
    Returns:
        List of VolumeData objects in time order
    """
    loader = TimeSeriesDicomLoader()
    return loader.load_series(parent_folder, sort_mode=sort_mode, callback=callback)


class TimeSeriesOrderDialog:
    """
    Dialog for manually reordering timepoint folders.
    
    Usage:
        dialog = TimeSeriesOrderDialog(parent_widget)
        ordered_folders = dialog.get_order(folder_list)
    """
    
    @staticmethod
    def get_order(parent_widget, folder_list: List[Tuple[int, str, str]]) -> Optional[List[str]]:
        """
        Show dialog to manually reorder folders.
        
        Args:
            parent_widget: Parent Qt widget for the dialog
            folder_list: List of (index, name, path) tuples
            
        Returns:
            List of folder paths in user-selected order, or None if cancelled
        """
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
            QPushButton, QLabel, QDialogButtonBox, QListWidgetItem
        )
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle("Reorder Timepoints")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(300)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Drag items or use buttons to reorder timepoints:"))
        
        # List widget with drag-and-drop
        list_widget = QListWidget()
        list_widget.setDragDropMode(QListWidget.InternalMove)
        
        for idx, name, path in folder_list:
            item = QListWidgetItem(f"t={idx}: {name}")
            item.setData(Qt.UserRole, path)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Move buttons
        btn_layout = QHBoxLayout()
        
        up_btn = QPushButton("↑ Move Up")
        down_btn = QPushButton("↓ Move Down")
        
        def move_up():
            row = list_widget.currentRow()
            if row > 0:
                item = list_widget.takeItem(row)
                list_widget.insertItem(row - 1, item)
                list_widget.setCurrentRow(row - 1)
        
        def move_down():
            row = list_widget.currentRow()
            if row < list_widget.count() - 1:
                item = list_widget.takeItem(row)
                list_widget.insertItem(row + 1, item)
                list_widget.setCurrentRow(row + 1)
        
        up_btn.clicked.connect(move_up)
        down_btn.clicked.connect(move_down)
        
        btn_layout.addWidget(up_btn)
        btn_layout.addWidget(down_btn)
        layout.addLayout(btn_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Return paths in new order
            ordered = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                ordered.append(item.data(Qt.UserRole))
            return ordered
        
        return None
