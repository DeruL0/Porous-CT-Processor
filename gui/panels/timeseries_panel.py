"""
Time Series Panel for 4D CT pore tracking visualization.

Provides controls for:
- Timeline slider navigation
- Playback animation
- Pore volume history chart
- Compression status display
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QComboBox, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from typing import Optional, List, Callable
import numpy as np

from config import TRACKING_ANIMATION_FPS


class TimeSeriesPanel(QWidget):
    """
    Panel for 4D CT time series navigation and pore tracking visualization.
    
    Signals:
        timepoint_changed(int): Emitted when user selects a different timepoint
        pore_selected(int): Emitted when user selects a pore to view details
        animation_toggled(bool): Emitted when animation is started/stopped
    """
    
    timepoint_changed = pyqtSignal(int)
    pore_selected = pyqtSignal(int)
    animation_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._num_timepoints = 0
        self._current_timepoint = 0
        self._is_animating = False
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._advance_frame)
        
        self._time_series_pnm = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # === Timeline Group ===
        timeline_group = QGroupBox("Timeline Navigation")
        timeline_layout = QVBoxLayout(timeline_group)
        
        # Timepoint label
        self._timepoint_label = QLabel("Timepoint: 0 / 0")
        self._timepoint_label.setStyleSheet("font-weight: bold;")
        timeline_layout.addWidget(self._timepoint_label)
        
        # Timeline slider
        slider_layout = QHBoxLayout()
        self._timeline_slider = QSlider(Qt.Horizontal)
        self._timeline_slider.setMinimum(0)
        self._timeline_slider.setMaximum(0)
        self._timeline_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(QLabel("t=0"))
        slider_layout.addWidget(self._timeline_slider, stretch=1)
        self._max_label = QLabel("t=0")
        slider_layout.addWidget(self._max_label)
        timeline_layout.addLayout(slider_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.clicked.connect(self._prev_timepoint)
        playback_layout.addWidget(self._prev_btn)
        
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_animation)
        playback_layout.addWidget(self._play_btn)
        
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._next_timepoint)
        playback_layout.addWidget(self._next_btn)
        
        timeline_layout.addLayout(playback_layout)
        layout.addWidget(timeline_group)
        
        # === Summary Group ===
        summary_group = QGroupBox("Tracking Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self._summary_table = QTableWidget(4, 2)
        self._summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.verticalHeader().setVisible(False)
        self._summary_table.setMaximumHeight(130)
        
        # Initialize with default values
        self._set_summary_row(0, "Total Pores", "—")
        self._set_summary_row(1, "Active", "—")
        self._set_summary_row(2, "Compressed", "—")
        self._set_summary_row(3, "Avg. Retention", "—")
        
        summary_layout.addWidget(self._summary_table)
        self._summary_group = summary_group
        layout.addWidget(summary_group)
        
        # === Pore List Group ===
        pore_group = QGroupBox("Pore Tracking")
        pore_layout = QVBoxLayout(pore_group)
        
        # Filter dropdown
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Pores", "Active Only", "Compressed Only"])
        self._filter_combo.currentIndexChanged.connect(self._update_pore_list)
        filter_layout.addWidget(self._filter_combo, stretch=1)
        pore_layout.addLayout(filter_layout)
        
        # Pore table
        self._pore_table = QTableWidget()
        self._pore_table.setColumnCount(4)
        self._pore_table.setHorizontalHeaderLabels(["ID", "Status", "Vol (t=0)", "Vol (now)"])
        self._pore_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._pore_table.verticalHeader().setVisible(False)
        self._pore_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._pore_table.setSelectionMode(QTableWidget.SingleSelection)
        self._pore_table.cellClicked.connect(self._on_pore_clicked)
        pore_layout.addWidget(self._pore_table)
        
        self._pore_group = pore_group
        layout.addWidget(pore_group, stretch=1)
        
        # === Export Button ===
        self._export_btn = QPushButton("📊 Export Volume History (CSV)")
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)
        
        # Initial state: disabled
        self.setEnabled(False)
    
    def set_volume_only_mode(self, num_timepoints: int, folder_names: Optional[List[str]] = None) -> None:
        """
        Enable timeline for volume-only navigation (before PNM tracking).
        
        Args:
            num_timepoints: Number of timepoints loaded
            folder_names: Optional list of folder names for each timepoint
        """
        self._time_series_pnm = None
        self._num_timepoints = num_timepoints
        self._current_timepoint = 0
        self._folder_names = folder_names or [f"t={i}" for i in range(num_timepoints)]
        
        # Update UI
        self._timeline_slider.setMaximum(max(0, num_timepoints - 1))
        self._timeline_slider.setValue(0)
        self._max_label.setText(f"t={num_timepoints - 1}")
        self._update_timepoint_label()
        
        # Hide tracking-specific UI since no PNM yet
        self._summary_group.setTitle("Tracking Summary (Run 'Track 4D Pores' first)")
        for i in range(4):
            self._set_summary_row(i, ["Total Pores", "Active", "Compressed", "Avg. Retention"][i], "—")
        
        self._pore_table.setRowCount(0)
        self._export_btn.setEnabled(False)
        
        # Enable timeline controls
        self.setEnabled(True)
    
    def _update_timepoint_label(self):
        """Update the timepoint label with folder name if available."""
        if hasattr(self, '_folder_names') and self._folder_names:
            name = self._folder_names[self._current_timepoint] if self._current_timepoint < len(self._folder_names) else ""
            self._timepoint_label.setText(f"Timepoint: {self._current_timepoint} / {self._num_timepoints - 1} ({name})")
        else:
            self._timepoint_label.setText(f"Timepoint: {self._current_timepoint} / {self._num_timepoints - 1}")
    
    def set_time_series(self, time_series_pnm) -> None:
        """
        Set the time series data for display.
        
        Args:
            time_series_pnm: TimeSeriesPNM object with tracking results
        """
        self._time_series_pnm = time_series_pnm
        self._num_timepoints = time_series_pnm.num_timepoints
        self._current_timepoint = 0
        
        # Update UI
        self._timeline_slider.setMaximum(max(0, self._num_timepoints - 1))
        self._timeline_slider.setValue(0)
        self._max_label.setText(f"t={self._num_timepoints - 1}")
        self._update_timepoint_label()
        
        # Update summary
        self._summary_group.setTitle("Tracking Summary")
        summary = time_series_pnm.get_summary()
        self._set_summary_row(0, "Total Pores", str(summary.get('reference_pores', 0)))
        self._set_summary_row(1, "Active", str(summary.get('active_pores', 0)))
        self._set_summary_row(2, "Compressed", str(summary.get('compressed_pores', 0)))
        self._set_summary_row(3, "Avg. Retention", f"{summary.get('avg_volume_retention', 0):.1%}")
        
        # Update pore list
        self._update_pore_list()
        
        self.setEnabled(True)
        self._export_btn.setEnabled(True)
    
    def _set_summary_row(self, row: int, name: str, value: str):
        """Set a row in the summary table."""
        self._summary_table.setItem(row, 0, QTableWidgetItem(name))
        self._summary_table.setItem(row, 1, QTableWidgetItem(value))
    
    def _update_pore_list(self):
        """Update the pore tracking table based on filter."""
        if self._time_series_pnm is None:
            return
        
        tracking = self._time_series_pnm.tracking
        filter_mode = self._filter_combo.currentIndex()
        
        # Get pore IDs based on filter
        if filter_mode == 0:  # All
            pore_ids = tracking.reference_ids
        elif filter_mode == 1:  # Active only
            pore_ids = tracking.get_active_pore_ids(self._current_timepoint)
        else:  # Compressed only
            pore_ids = tracking.get_compressed_pore_ids(self._current_timepoint)
        
        self._pore_table.setRowCount(len(pore_ids))
        
        for row, pore_id in enumerate(pore_ids):
            volumes = tracking.volume_history.get(pore_id, [])
            statuses = tracking.status_history.get(pore_id, [])
            
            vol_t0 = volumes[0] if volumes else 0
            vol_now = volumes[self._current_timepoint] if self._current_timepoint < len(volumes) else 0
            status = statuses[self._current_timepoint].value if self._current_timepoint < len(statuses) else "unknown"
            
            self._pore_table.setItem(row, 0, QTableWidgetItem(str(pore_id)))
            
            status_item = QTableWidgetItem(status)
            if status == "compressed":
                status_item.setBackground(QColor(255, 200, 200))
            else:
                status_item.setBackground(QColor(200, 255, 200))
            self._pore_table.setItem(row, 1, status_item)
            
            self._pore_table.setItem(row, 2, QTableWidgetItem(f"{vol_t0:.0f}"))
            self._pore_table.setItem(row, 3, QTableWidgetItem(f"{vol_now:.0f}"))
    
    def _on_slider_changed(self, value: int):
        """Handle timeline slider value change."""
        self._current_timepoint = value
        self._update_timepoint_label()
        self._update_pore_list()
        self.timepoint_changed.emit(value)
    
    def _prev_timepoint(self):
        """Go to previous timepoint."""
        if self._current_timepoint > 0:
            self._timeline_slider.setValue(self._current_timepoint - 1)
    
    def _next_timepoint(self):
        """Go to next timepoint."""
        if self._current_timepoint < self._num_timepoints - 1:
            self._timeline_slider.setValue(self._current_timepoint + 1)
    
    def _toggle_animation(self):
        """Start/stop animation playback."""
        self._is_animating = not self._is_animating
        
        if self._is_animating:
            self._play_btn.setText("⏸ Pause")
            interval_ms = int(1000 / TRACKING_ANIMATION_FPS)
            self._animation_timer.start(interval_ms)
        else:
            self._play_btn.setText("▶ Play")
            self._animation_timer.stop()
        
        self.animation_toggled.emit(self._is_animating)
    
    def _advance_frame(self):
        """Advance to next frame in animation."""
        next_frame = self._current_timepoint + 1
        if next_frame >= self._num_timepoints:
            next_frame = 0  # Loop back
        self._timeline_slider.setValue(next_frame)
    
    def _on_pore_clicked(self, row: int, col: int):
        """Handle pore table row click."""
        item = self._pore_table.item(row, 0)
        if item:
            pore_id = int(item.text())
            self.pore_selected.emit(pore_id)
    
    def _export_csv(self):
        """Export volume history to CSV."""
        if self._time_series_pnm is None:
            return
        
        from PyQt5.QtWidgets import QFileDialog
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Volume History", 
            "pore_volumes.csv", 
            "CSV Files (*.csv)"
        )
        
        if filepath:
            from processors.pnm_tracker import PNMTracker
            tracker = PNMTracker()
            tracker.time_series = self._time_series_pnm
            tracker.export_volume_csv(filepath)
    
    def get_current_timepoint(self) -> int:
        """Get the currently selected timepoint index."""
        return self._current_timepoint
    
    def reset(self):
        """Reset the panel to initial state."""
        self._time_series_pnm = None
        self._num_timepoints = 0
        self._current_timepoint = 0
        
        if self._is_animating:
            self._toggle_animation()
        
        self._timeline_slider.setMaximum(0)
        self._timeline_slider.setValue(0)
        self._timepoint_label.setText("Timepoint: 0 / 0")
        self._max_label.setText("t=0")
        
        for i in range(4):
            self._set_summary_row(i, ["Total Pores", "Active", "Compressed", "Avg. Retention"][i], "—")
        
        self._pore_table.setRowCount(0)
        self.setEnabled(False)
        self._export_btn.setEnabled(False)
