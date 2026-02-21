"""
Visualization Mode Panel for selecting rendering modes.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QFrame
from PyQt5.QtCore import pyqtSignal

from gui.styles import PANEL_TITLE_STYLE
from gui.ui_constants import apply_group_layout, make_description_label, set_primary_button_policy


class Separator(QFrame):
    """Simple horizontal line separator"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VisualizationModePanel(QGroupBox):
    """
    Panel for selecting the Visualization Mode (Volume, Slices, Iso).
    """
    volume_clicked = pyqtSignal()
    slices_clicked = pyqtSignal()
    iso_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    reset_camera_clicked = pyqtSignal()

    def __init__(self, title: str = "🖼️ Analysis Modes"):
        super().__init__()
        self.custom_title = title
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        apply_group_layout(layout)
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(title_lbl)
        layout.addWidget(title_lbl)
        
        self._add_button(layout, "📊 Volume Rendering", self.volume_clicked)
        self._add_button(layout, "🔳 Orthogonal Slices", self.slices_clicked)
        self._add_button(layout, "🏔️ Isosurface (Solid/Pore)", self.iso_clicked)
        layout.addWidget(Separator())
        self._add_button(layout, "🗑️ Clear View", self.clear_clicked, min_height=35)
        self._add_button(layout, "🎥 Reset Camera", self.reset_camera_clicked, min_height=35)
        self.setLayout(layout)

    def _add_button(self, layout, text, signal, min_height=40):
        btn = QPushButton(text)
        btn.setMinimumHeight(min_height)
        set_primary_button_policy(btn)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)
