"""
Clip Plane Panel for interactive volume clipping.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from typing import Dict, Any

from gui.styles import PANEL_TITLE_STYLE
from gui.ui_constants import apply_group_layout, make_description_label


class ClipPlanePanel(QGroupBox):
    """
    Panel for interactive clip plane controls.
    """
    clip_changed = pyqtSignal()
    clip_toggled = pyqtSignal(bool)

    def __init__(self, title: str = "✂️ Clip Planes"):
        super().__init__()
        self.custom_title = title
        self._enabled = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        apply_group_layout(layout)
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(title_lbl)
        layout.addWidget(title_lbl)

        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable Clipping")
        self.enable_checkbox.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.enable_checkbox)

        # X Clip
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 100)
        self.slider_x.setValue(50)
        self.slider_x.valueChanged.connect(self.clip_changed.emit)
        x_layout.addWidget(self.slider_x)
        self.invert_x = QCheckBox("Invert")
        self.invert_x.stateChanged.connect(self.clip_changed.emit)
        x_layout.addWidget(self.invert_x)
        layout.addLayout(x_layout)

        # Y Clip
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 100)
        self.slider_y.setValue(50)
        self.slider_y.valueChanged.connect(self.clip_changed.emit)
        y_layout.addWidget(self.slider_y)
        self.invert_y = QCheckBox("Invert")
        self.invert_y.stateChanged.connect(self.clip_changed.emit)
        y_layout.addWidget(self.invert_y)
        layout.addLayout(y_layout)

        # Z Clip
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 100)
        self.slider_z.setValue(50)
        self.slider_z.valueChanged.connect(self.clip_changed.emit)
        z_layout.addWidget(self.slider_z)
        self.invert_z = QCheckBox("Invert")
        self.invert_z.stateChanged.connect(self.clip_changed.emit)
        z_layout.addWidget(self.invert_z)
        layout.addLayout(z_layout)

        self.setLayout(layout)
        self._update_slider_state()

    def _on_toggle(self, state):
        self._enabled = (state == Qt.Checked)
        self._update_slider_state()
        self.clip_toggled.emit(self._enabled)

    def _update_slider_state(self):
        enabled = self._enabled
        self.slider_x.setEnabled(enabled)
        self.slider_y.setEnabled(enabled)
        self.slider_z.setEnabled(enabled)
        self.invert_x.setEnabled(enabled)
        self.invert_y.setEnabled(enabled)
        self.invert_z.setEnabled(enabled)

    def get_clip_values(self) -> Dict[str, Any]:
        return {
            'enabled': self._enabled,
            'x': self.slider_x.value() / 100.0,
            'y': self.slider_y.value() / 100.0,
            'z': self.slider_z.value() / 100.0,
            'invert_x': self.invert_x.isChecked(),
            'invert_y': self.invert_y.isChecked(),
            'invert_z': self.invert_z.isChecked(),
        }
