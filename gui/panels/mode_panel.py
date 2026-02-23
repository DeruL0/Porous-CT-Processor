"""
Visualization Mode Panel for selecting rendering modes.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from gui.styles import PANEL_TITLE_STYLE
from gui.ui_constants import apply_group_layout, make_description_label, set_primary_button_policy


class Separator(QFrame):
    """Simple horizontal line separator."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VisualizationModePanel(QGroupBox):
    """
    Panel for selecting visualization modes and additive overlays.
    """

    volume_clicked = pyqtSignal()
    slices_clicked = pyqtSignal()
    iso_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    reset_camera_clicked = pyqtSignal()

    overlay_layer_requested = pyqtSignal(str, str, float)  # layer_key, source_name, opacity(0-1)
    overlays_cleared = pyqtSignal()

    def __init__(self, title: str = "Analysis Modes"):
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

        self._add_button(layout, "Volume Rendering", self.volume_clicked)
        self._add_button(layout, "Orthogonal Slices", self.slices_clicked)
        self._add_button(layout, "Isosurface (Solid/Pore)", self.iso_clicked)

        layout.addWidget(Separator())
        self._init_overlay_section(layout)
        layout.addWidget(Separator())

        self._add_button(layout, "Clear View", self.clear_clicked, min_height=35)
        self._add_button(layout, "Reset Camera", self.reset_camera_clicked, min_height=35)
        self.setLayout(layout)

    def _init_overlay_section(self, parent_layout: QVBoxLayout) -> None:
        section_title = QLabel("Overlay on Current View")
        section_title.setStyleSheet("font-weight: bold;")
        make_description_label(section_title)
        parent_layout.addWidget(section_title)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Volume Source:"))
        self.overlay_source_combo = QComboBox()
        self.overlay_source_combo.addItems(["Raw CT", "Segmented"])
        source_row.addWidget(self.overlay_source_combo, stretch=1)
        parent_layout.addLayout(source_row)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self.overlay_opacity_slider = QSlider(Qt.Horizontal)
        self.overlay_opacity_slider.setRange(5, 100)
        self.overlay_opacity_slider.setValue(35)
        self.overlay_opacity_label = QLabel("35%")
        self.overlay_opacity_slider.valueChanged.connect(
            lambda value: self.overlay_opacity_label.setText(f"{value}%")
        )
        opacity_row.addWidget(self.overlay_opacity_slider, stretch=1)
        opacity_row.addWidget(self.overlay_opacity_label)
        parent_layout.addLayout(opacity_row)

        grid = QGridLayout()
        self._add_overlay_button(grid, 0, 0, "+ Layer: Slices", "slices")
        self._add_overlay_button(grid, 0, 1, "+ Layer: Iso", "iso")
        self._add_overlay_button(grid, 1, 0, "+ Layer: Volume", "volume")
        self._add_overlay_button(grid, 1, 1, "+ Layer: PNM Mesh", "pnm_mesh")
        parent_layout.addLayout(grid)

        clear_btn = QPushButton("Clear Overlays")
        clear_btn.setMinimumHeight(34)
        set_primary_button_policy(clear_btn)
        clear_btn.clicked.connect(self.overlays_cleared.emit)
        parent_layout.addWidget(clear_btn)

    def _add_overlay_button(self, layout: QGridLayout, row: int, col: int, text: str, layer_key: str) -> None:
        btn = QPushButton(text)
        btn.setMinimumHeight(34)
        set_primary_button_policy(btn)
        btn.clicked.connect(lambda _=False, key=layer_key: self._emit_overlay_request(key))
        layout.addWidget(btn, row, col)

    def _emit_overlay_request(self, layer_key: str) -> None:
        source_name = self.overlay_source_combo.currentText()
        opacity = float(self.overlay_opacity_slider.value()) / 100.0
        self.overlay_layer_requested.emit(layer_key, source_name, opacity)

    def _add_button(self, layout, text, signal, min_height=40):
        btn = QPushButton(text)
        btn.setMinimumHeight(min_height)
        set_primary_button_policy(btn)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)
