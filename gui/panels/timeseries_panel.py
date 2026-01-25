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



class TimeSeriesControlPanel(QGroupBox):
    """
    Panel for 4D CT timeline navigation and playback.
    Placed on the left sidebar.
    """
    timepoint_changed = pyqtSignal(int)
    animation_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("⏱ Time Navigation", parent)
        self._num_timepoints = 0
        self._current_timepoint = 0
        self._is_animating = False
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._advance_frame)
        self._folder_names = []
        
        self._setup_ui()
        self.setEnabled(False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Timepoint label
        self._timepoint_label = QLabel("Timepoint: 0 / 0")
        self._timepoint_label.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(self._timepoint_label)
        
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
        layout.addLayout(slider_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setMinimumHeight(35)
        self._prev_btn.clicked.connect(self._prev_timepoint)
        playback_layout.addWidget(self._prev_btn)
        
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setMinimumHeight(35)
        self._play_btn.clicked.connect(self._toggle_animation)
        playback_layout.addWidget(self._play_btn)
        
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setMinimumHeight(35)
        self._next_btn.clicked.connect(self._next_timepoint)
        playback_layout.addWidget(self._next_btn)
        
        layout.addLayout(playback_layout)

    def set_range(self, num_timepoints: int, folder_names: Optional[List[str]] = None):
        self._num_timepoints = num_timepoints
        self._folder_names = folder_names or [f"t={i}" for i in range(num_timepoints)]
        self._timeline_slider.setMaximum(max(0, num_timepoints - 1))
        self._timeline_slider.setValue(0)
        self._max_label.setText(f"t={num_timepoints - 1}")
        self._update_timepoint_label()
        self.setEnabled(True)

    def _update_timepoint_label(self):
        if self._folder_names and self._current_timepoint < len(self._folder_names):
            name = self._folder_names[self._current_timepoint]
            self._timepoint_label.setText(f"Timepoint: {self._current_timepoint} / {self._num_timepoints - 1}\n({name})")
        else:
            self._timepoint_label.setText(f"Timepoint: {self._current_timepoint} / {self._num_timepoints - 1}")

    def _on_slider_changed(self, value: int):
        self._current_timepoint = value
        self._update_timepoint_label()
        self.timepoint_changed.emit(value)

    def _prev_timepoint(self):
        if self._current_timepoint > 0:
            self._timeline_slider.setValue(self._current_timepoint - 1)

    def _next_timepoint(self):
        if self._current_timepoint < self._num_timepoints - 1:
            self._timeline_slider.setValue(self._current_timepoint + 1)

    def _toggle_animation(self):
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
        next_frame = self._current_timepoint + 1
        if next_frame >= self._num_timepoints:
            next_frame = 0
        self._timeline_slider.setValue(next_frame)

    def reset(self):
        self._animation_timer.stop()
        self._is_animating = False
        self._play_btn.setText("▶ Play")
        self._timeline_slider.setMaximum(0)
        self._timeline_slider.setValue(0)
        self._timepoint_label.setText("Timepoint: 0 / 0")
        self.setEnabled(False)


class TrackingAnalysisPanel(QWidget):
    """
    Panel for Tracking summary and detailed pore list.
    Placed on the right sidebar.
    """
    pore_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time_series_pnm = None
        self._current_timepoint = 0
        self._setup_ui()
        self.setEnabled(False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # === Summary Group ===
        self._summary_group = QGroupBox("📊 Tracking Summary")
        summary_layout = QVBoxLayout(self._summary_group)
        
        self._summary_table = QTableWidget(4, 2)
        self._summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.verticalHeader().setVisible(False)
        self._summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._summary_table.setMaximumHeight(200)

        
        self._set_summary_row(0, "Total Pores", "—")
        self._set_summary_row(1, "Active", "—")
        self._set_summary_row(2, "Compressed", "—")
        self._set_summary_row(3, "Avg. Retention", "—")
        
        summary_layout.addWidget(self._summary_table)
        layout.addWidget(self._summary_group)
        
        # === Pore List Group ===
        pore_group = QGroupBox("🔍 Pore Detailed Tracking")
        pore_layout = QVBoxLayout(pore_group)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Pores", "Active Only", "Compressed Only"])
        self._filter_combo.currentIndexChanged.connect(self.update_view)
        filter_layout.addWidget(self._filter_combo, stretch=1)
        pore_layout.addLayout(filter_layout)
        
        self._pore_table = QTableWidget()
        self._pore_table.setColumnCount(4)
        self._pore_table.setHorizontalHeaderLabels(["ID", "Status", "Vol (t=0)", "Vol (now)"])
        self._pore_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._pore_table.verticalHeader().setVisible(False)
        self._pore_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._pore_table.setSelectionMode(QTableWidget.SingleSelection)
        self._pore_table.cellClicked.connect(self._on_pore_clicked)
        pore_layout.addWidget(self._pore_table)
        
        layout.addWidget(pore_group)
        layout.addStretch()

        
        # Export Button
        self._export_btn = QPushButton("💾 Export Volume History (CSV)")
        self._export_btn.setMinimumHeight(35)
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400):
        """Dynamically adjust table height to fit contents."""
        table.doItemsLayout()
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        height += 5
        
        clamped_height = max(100, min(height, max_height))
        table.setFixedHeight(clamped_height)
        
        if height > max_height:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def set_time_series(self, time_series_pnm):
        self._time_series_pnm = time_series_pnm
        self._current_timepoint = 0
        summary = time_series_pnm.get_summary()
        self._set_summary_row(0, "Total Pores", str(summary.get('reference_pores', 0)))
        self._set_summary_row(1, "Active", str(summary.get('active_pores', 0)))
        self._set_summary_row(2, "Compressed", str(summary.get('compressed_pores', 0)))
        self._set_summary_row(3, "Avg. Retention", f"{summary.get('avg_volume_retention', 0):.1%}")
        
        self._adjust_table_height(self._summary_table, max_height=250)



        
        self.update_view()
        self.setEnabled(True)
        self._export_btn.setEnabled(True)

    def set_timepoint(self, index: int):
        self._current_timepoint = index
        self.update_view()

    def update_view(self):
        if self._time_series_pnm is None:
            return
        
        tracking = self._time_series_pnm.tracking
        filter_mode = self._filter_combo.currentIndex()
        
        if filter_mode == 0:
            pore_ids = tracking.reference_ids
        elif filter_mode == 1:
            pore_ids = tracking.get_active_pore_ids(self._current_timepoint)
        else:
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
            
        self._adjust_table_height(self._pore_table, max_height=500)



    def _set_summary_row(self, row: int, name: str, value: str):
        self._summary_table.setItem(row, 0, QTableWidgetItem(name))
        self._summary_table.setItem(row, 1, QTableWidgetItem(value))

    def _on_pore_clicked(self, row: int, col: int):
        item = self._pore_table.item(row, 0)
        if item:
            self.pore_selected.emit(int(item.text()))

    def _export_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Volume History", "pore_volumes.csv", "CSV Files (*.csv)")
        if filepath:
            from processors.pnm_tracker import PNMTracker
            tracker = PNMTracker()
            tracker.time_series = self._time_series_pnm
            tracker.export_volume_csv(filepath)

    def reset(self):
        self._time_series_pnm = None
        self._current_timepoint = 0
        self._pore_table.setRowCount(0)
        for i in range(4):
            self._set_summary_row(i, ["Total Pores", "Active", "Compressed", "Avg. Retention"][i], "—")
        self.setEnabled(False)
        self._export_btn.setEnabled(False)

