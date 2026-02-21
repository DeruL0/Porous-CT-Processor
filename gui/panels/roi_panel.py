"""
ROI Panel for interactive Region of Interest selection.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from gui.styles import PANEL_TITLE_STYLE
from gui.ui_constants import apply_group_layout, make_description_label, set_primary_button_policy


class ROIPanel(QGroupBox):
    """
    Panel for interactive ROI selection.
    All shapes are controlled by one box widget. Cylinder and sphere are inscribed.
    """

    roi_toggled = pyqtSignal(bool)
    apply_roi = pyqtSignal()
    reset_roi = pyqtSignal()
    shape_changed = pyqtSignal(str)

    def __init__(self, title: str = "ROI Selection"):
        super().__init__()
        self.custom_title = title
        self._enabled = False
        self._bounds = None
        self._shape = "box"
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        apply_group_layout(layout)

        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(title_lbl)
        layout.addWidget(title_lbl)

        shape_layout = QHBoxLayout()
        shape_layout.setSpacing(8)
        shape_layout.addWidget(QLabel("Shape:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Box", "Cylinder", "Sphere"])
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        shape_layout.addWidget(self.shape_combo)
        layout.addLayout(shape_layout)

        self.shape_desc = QLabel("Extract rectangular region.")
        self.shape_desc.setStyleSheet("color: gray; font-size: 10px;")
        make_description_label(self.shape_desc)
        layout.addWidget(self.shape_desc)

        self.enable_checkbox = QCheckBox("Enable ROI Selection")
        self.enable_checkbox.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.enable_checkbox)

        self.bounds_label = QLabel("Bounds: Not selected")
        make_description_label(self.bounds_label)
        layout.addWidget(self.bounds_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.apply_btn = QPushButton("Apply ROI")
        self.apply_btn.setEnabled(False)
        set_primary_button_policy(self.apply_btn)
        self.apply_btn.clicked.connect(self.apply_roi.emit)
        btn_layout.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("Reset")
        set_primary_button_policy(self.reset_btn)
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _on_shape_changed(self, text: str) -> None:
        self._shape = text.lower()
        descriptions = {
            "box": "Extract rectangular region.",
            "cylinder": "Extract cylinder inscribed in the box (YZ plane).",
            "sphere": "Extract sphere inscribed in the box.",
        }
        self.shape_desc.setText(descriptions.get(self._shape, ""))
        self._bounds = None
        self.bounds_label.setText("Bounds: Not selected")
        self.apply_btn.setEnabled(False)
        self.shape_changed.emit(self._shape)

    def _on_toggle(self, state: int) -> None:
        self._enabled = (state == Qt.Checked)
        self.roi_toggled.emit(self._enabled)
        if not self._enabled:
            self._bounds = None
            self.bounds_label.setText("Bounds: Not selected")
            self.apply_btn.setEnabled(False)

    def _on_reset(self) -> None:
        self.enable_checkbox.setChecked(False)
        self._bounds = None
        self.bounds_label.setText("Bounds: Not selected")
        self.apply_btn.setEnabled(False)
        self.reset_roi.emit()

    def update_bounds(self, bounds: tuple) -> None:
        self._bounds = bounds
        if not bounds:
            self.bounds_label.setText("Bounds: Not selected")
            self.apply_btn.setEnabled(False)
            return

        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]
        shape_info = ""
        if self._shape == "sphere":
            rx, ry, rz = size_x / 2, size_y / 2, size_z / 2
            shape_info = f"\nEllipsoid Rx:{rx:.1f} Ry:{ry:.1f} Rz:{rz:.1f}"
        elif self._shape == "cylinder":
            ry, rz = size_y / 2, size_z / 2
            shape_info = f"\nCylinder Ry:{ry:.1f} Rz:{rz:.1f} H:{size_x:.1f}"

        self.bounds_label.setText(
            f"X: {bounds[0]:.1f} - {bounds[1]:.1f}\n"
            f"Y: {bounds[2]:.1f} - {bounds[3]:.1f}\n"
            f"Z: {bounds[4]:.1f} - {bounds[5]:.1f}{shape_info}"
        )
        self.apply_btn.setEnabled(True)

    def get_shape(self) -> str:
        return self._shape

    def get_bounds(self) -> tuple:
        return self._bounds

    def is_enabled(self) -> bool:
        return self._enabled
