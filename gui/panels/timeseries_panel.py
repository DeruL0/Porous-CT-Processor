"""
Time series panels for 4D CT tracking workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import TRACKING_ANIMATION_FPS
from gui.ui_constants import (
    TABLE_MIN_HEIGHT,
    apply_group_layout,
    apply_panel_layout,
    make_description_label,
    set_primary_button_policy,
)


class TimeSeriesControlPanel(QGroupBox):
    """
    Left-side timeline controls for 4DCT navigation.
    """

    timepoint_changed = pyqtSignal(int)
    animation_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Time Navigation", parent)
        self._num_timepoints = 0
        self._current_timepoint = 0
        self._is_animating = False
        self._folder_names: List[str] = []

        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._advance_frame)

        self._setup_ui()
        self.setEnabled(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        apply_group_layout(layout)

        self._timepoint_label = QLabel("Timepoint: 0 / 0")
        self._timepoint_label.setStyleSheet("font-weight: bold; color: #3498db;")
        make_description_label(self._timepoint_label)
        layout.addWidget(self._timepoint_label)

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

        playback_layout = QHBoxLayout()
        self._prev_btn = QPushButton("Prev")
        self._prev_btn.setMinimumHeight(32)
        set_primary_button_policy(self._prev_btn)
        self._prev_btn.clicked.connect(self._prev_timepoint)
        playback_layout.addWidget(self._prev_btn)

        self._play_btn = QPushButton("Play")
        self._play_btn.setMinimumHeight(32)
        set_primary_button_policy(self._play_btn)
        self._play_btn.clicked.connect(self._toggle_animation)
        playback_layout.addWidget(self._play_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setMinimumHeight(32)
        set_primary_button_policy(self._next_btn)
        self._next_btn.clicked.connect(self._next_timepoint)
        playback_layout.addWidget(self._next_btn)

        layout.addLayout(playback_layout)

    def set_range(self, num_timepoints: int, folder_names: Optional[List[str]] = None) -> None:
        self._num_timepoints = int(max(num_timepoints, 0))
        self._folder_names = folder_names or [f"t={i}" for i in range(self._num_timepoints)]
        self._timeline_slider.setMaximum(max(0, self._num_timepoints - 1))
        self._timeline_slider.setValue(0)
        self._max_label.setText(f"t={max(0, self._num_timepoints - 1)}")
        self._update_timepoint_label()
        self.setEnabled(True)

    def _update_timepoint_label(self) -> None:
        if self._folder_names and self._current_timepoint < len(self._folder_names):
            name = self._folder_names[self._current_timepoint]
            self._timepoint_label.setText(
                f"Timepoint: {self._current_timepoint} / {max(0, self._num_timepoints - 1)}\n({name})"
            )
        else:
            self._timepoint_label.setText(
                f"Timepoint: {self._current_timepoint} / {max(0, self._num_timepoints - 1)}"
            )

    def _on_slider_changed(self, value: int) -> None:
        self._current_timepoint = int(value)
        self._update_timepoint_label()
        self.timepoint_changed.emit(int(value))

    def _prev_timepoint(self) -> None:
        if self._current_timepoint > 0:
            self._timeline_slider.setValue(self._current_timepoint - 1)

    def _next_timepoint(self) -> None:
        if self._current_timepoint < self._num_timepoints - 1:
            self._timeline_slider.setValue(self._current_timepoint + 1)

    def _toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._play_btn.setText("Pause")
            interval_ms = int(1000 / max(TRACKING_ANIMATION_FPS, 1))
            self._animation_timer.start(interval_ms)
        else:
            self._play_btn.setText("Play")
            self._animation_timer.stop()
        self.animation_toggled.emit(self._is_animating)

    def _advance_frame(self) -> None:
        next_frame = self._current_timepoint + 1
        if next_frame >= self._num_timepoints:
            next_frame = 0
        self._timeline_slider.setValue(next_frame)

    def reset(self) -> None:
        self._animation_timer.stop()
        self._is_animating = False
        self._play_btn.setText("Play")
        self._timeline_slider.setMaximum(0)
        self._timeline_slider.setValue(0)
        self._timepoint_label.setText("Timepoint: 0 / 0")
        self.setEnabled(False)


class TrackingAnalysisPanel(QWidget):
    """
    Right-side tracking analysis panel.
    """

    pore_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time_series_pnm = None
        self._current_timepoint = 0
        self._eval_by_time_and_pore: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._has_eval = False
        self._setup_ui()
        self.setEnabled(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        self._summary_group = QGroupBox("Tracking Summary")
        summary_layout = QVBoxLayout(self._summary_group)
        apply_group_layout(summary_layout)

        self._summary_table = QTableWidget(4, 2)
        self._summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._summary_table.verticalHeader().setVisible(False)
        self._summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._summary_table.setMinimumHeight(TABLE_MIN_HEIGHT)
        self._summary_table.setMaximumHeight(260)

        self._set_summary_row(0, "Total Pores", "--")
        self._set_summary_row(1, "Active", "--")
        self._set_summary_row(2, "Compressed", "--")
        self._set_summary_row(3, "Avg. Retention", "--")
        summary_layout.addWidget(self._summary_table)
        layout.addWidget(self._summary_group)

        pore_group = QGroupBox("Pore Detailed Tracking")
        pore_layout = QVBoxLayout(pore_group)
        apply_group_layout(pore_layout)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Pores", "Active Only", "Compressed Only"])
        self._filter_combo.currentIndexChanged.connect(self.update_view)
        filter_layout.addWidget(self._filter_combo, stretch=1)
        pore_layout.addLayout(filter_layout)

        self._pore_table = QTableWidget()
        self._pore_table.setColumnCount(5)
        self._pore_table.setHorizontalHeaderLabels(["ID", "Status", "Vol (t=0)", "Vol (now)", "Eval"])
        self._pore_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._pore_table.verticalHeader().setVisible(False)
        self._pore_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._pore_table.setSelectionMode(QTableWidget.SingleSelection)
        self._pore_table.setMinimumHeight(TABLE_MIN_HEIGHT)
        self._pore_table.cellClicked.connect(self._on_pore_clicked)
        pore_layout.addWidget(self._pore_table)

        layout.addWidget(pore_group)
        layout.addStretch()

        self._export_btn = QPushButton("Export Volume History (CSV)")
        self._export_btn.setMinimumHeight(32)
        set_primary_button_policy(self._export_btn)
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        layout.addWidget(self._export_btn)

    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400) -> None:
        table.doItemsLayout()
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        height += 6
        clamped_height = max(TABLE_MIN_HEIGHT, min(height, max_height))
        table.setMinimumHeight(clamped_height)
        table.setMaximumHeight(max_height)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if height > max_height else Qt.ScrollBarAlwaysOff
        )

    def _set_summary_row(self, row: int, name: str, value: str) -> None:
        self._summary_table.setItem(row, 0, QTableWidgetItem(name))
        self._summary_table.setItem(row, 1, QTableWidgetItem(value))

    def _index_evaluation(self) -> None:
        self._eval_by_time_and_pore = {}
        self._has_eval = False
        if self._time_series_pnm is None:
            return

        tracking = getattr(self._time_series_pnm, "tracking", None)
        eval_report = getattr(tracking, "evaluation", {}) if tracking is not None else {}
        if not isinstance(eval_report, dict) or not bool(eval_report.get("available", False)):
            return

        steps = eval_report.get("steps", [])
        if not isinstance(steps, list):
            return

        for step in steps:
            if not isinstance(step, dict):
                continue
            t = step.get("time_index")
            if not isinstance(t, int):
                continue
            tracking_eval = step.get("tracking", {})
            one_step: Dict[int, Dict[str, Any]] = {}

            if isinstance(tracking_eval, dict):
                per_ref = tracking_eval.get("per_reference", {})
                if isinstance(per_ref, dict):
                    for key, detail in per_ref.items():
                        if not isinstance(detail, dict):
                            continue
                        try:
                            ref_id = int(key)
                        except Exception:
                            ref_id = int(detail.get("ref_pred_id", -1))
                        if ref_id <= 0:
                            continue
                        one_step[ref_id] = detail

            # Fallback for t=0: only instance mapping is available.
            if not one_step:
                mapping = step.get("mapping", {})
                if isinstance(mapping, dict):
                    pred_to_gt = mapping.get("pred_to_gt", {})
                    if isinstance(pred_to_gt, dict):
                        for key, gt_id in pred_to_gt.items():
                            try:
                                ref_id = int(key)
                                gt_int = int(gt_id)
                            except Exception:
                                continue
                            one_step[ref_id] = {
                                "ref_pred_id": ref_id,
                                "ref_gt_id": gt_int,
                                "present_in_gt": True,
                                "matched_pred_id": ref_id,
                                "mapped_gt_id": gt_int,
                                "outcome": "mapped_t0",
                                "correct": True,
                            }

            if one_step:
                self._eval_by_time_and_pore[t] = one_step

        self._has_eval = len(self._eval_by_time_and_pore) > 0

    @staticmethod
    def _eval_label_and_color(eval_detail: Optional[Dict[str, Any]]) -> Tuple[str, Optional[QColor]]:
        if not isinstance(eval_detail, dict):
            return ("n/a", None)

        outcome = str(eval_detail.get("outcome", "unknown"))
        is_correct = bool(eval_detail.get("correct", False))
        if is_correct:
            if outcome == "correct_absent":
                return ("closure ok", QColor(212, 242, 220))
            return ("match ok", QColor(212, 242, 220))

        mapping: Dict[str, Tuple[str, QColor]] = {
            "mapped_t0": ("mapped", QColor(212, 242, 220)),
            "missed": ("missed", QColor(255, 224, 224)),
            "id_switched": ("id switch", QColor(255, 210, 200)),
            "unmatched_to_gt": ("no gt match", QColor(255, 235, 205)),
            "false_positive_alive": ("false alive", QColor(255, 224, 224)),
            "unknown": ("unknown", QColor(235, 235, 235)),
        }
        return mapping.get(outcome, (outcome, QColor(235, 235, 235)))

    def _colorize_row(self, row: int, color: Optional[QColor]) -> None:
        if color is None:
            return
        for col in range(self._pore_table.columnCount()):
            item = self._pore_table.item(row, col)
            if item is not None:
                item.setBackground(color)

    def set_time_series(self, time_series_pnm) -> None:
        self._time_series_pnm = time_series_pnm
        self._current_timepoint = 0
        self._index_evaluation()

        summary = time_series_pnm.get_summary()
        self._set_summary_row(0, "Total Pores", str(summary.get("reference_pores", 0)))
        self._set_summary_row(1, "Active", str(summary.get("active_pores", 0)))
        self._set_summary_row(2, "Compressed", str(summary.get("compressed_pores", 0)))
        self._set_summary_row(3, "Avg. Retention", f"{summary.get('avg_volume_retention', 0):.1%}")
        self._adjust_table_height(self._summary_table, max_height=260)

        self.update_view()
        self.setEnabled(True)
        self._export_btn.setEnabled(True)

    def set_timepoint(self, index: int) -> None:
        self._current_timepoint = int(index)
        self.update_view()

    def update_view(self) -> None:
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

        eval_for_time = self._eval_by_time_and_pore.get(self._current_timepoint, {}) if self._has_eval else {}

        self._pore_table.setRowCount(len(pore_ids))
        for row, pore_id in enumerate(pore_ids):
            volumes = tracking.volume_history.get(pore_id, [])
            statuses = tracking.status_history.get(pore_id, [])

            vol_t0 = float(volumes[0]) if volumes else 0.0
            vol_now = float(volumes[self._current_timepoint]) if self._current_timepoint < len(volumes) else 0.0
            status = statuses[self._current_timepoint].value if self._current_timepoint < len(statuses) else "unknown"

            self._pore_table.setItem(row, 0, QTableWidgetItem(str(pore_id)))
            status_item = QTableWidgetItem(status)
            self._pore_table.setItem(row, 1, status_item)
            self._pore_table.setItem(row, 2, QTableWidgetItem(f"{vol_t0:.0f}"))
            self._pore_table.setItem(row, 3, QTableWidgetItem(f"{vol_now:.0f}"))

            eval_detail = eval_for_time.get(int(pore_id)) if isinstance(eval_for_time, dict) else None
            eval_label, eval_color = self._eval_label_and_color(eval_detail)
            self._pore_table.setItem(row, 4, QTableWidgetItem(eval_label))

            if self._has_eval and eval_detail is not None:
                self._colorize_row(row, eval_color)
            else:
                fallback = QColor(255, 225, 225) if status == "compressed" else QColor(220, 245, 220)
                status_item.setBackground(fallback)

        self._adjust_table_height(self._pore_table, max_height=520)

    def _on_pore_clicked(self, row: int, _col: int) -> None:
        item = self._pore_table.item(row, 0)
        if item:
            self.pore_selected.emit(int(item.text()))

    def _export_csv(self) -> None:
        from PyQt5.QtWidgets import QFileDialog
        from processors.pnm_tracker import PNMTracker

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Volume History",
            "pore_volumes.csv",
            "CSV Files (*.csv)",
        )
        if not filepath or self._time_series_pnm is None:
            return

        tracker = PNMTracker()
        tracker.time_series = self._time_series_pnm
        tracker.export_volume_csv(filepath)

    def reset(self) -> None:
        self._time_series_pnm = None
        self._current_timepoint = 0
        self._eval_by_time_and_pore = {}
        self._has_eval = False
        self._pore_table.setRowCount(0)
        self._set_summary_row(0, "Total Pores", "--")
        self._set_summary_row(1, "Active", "--")
        self._set_summary_row(2, "Compressed", "--")
        self._set_summary_row(3, "Avg. Retention", "--")
        self.setEnabled(False)
        self._export_btn.setEnabled(False)
