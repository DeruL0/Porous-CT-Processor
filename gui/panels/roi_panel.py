"""
ROI Panel for interactive Region of Interest selection.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QPushButton, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from gui.styles import PANEL_TITLE_STYLE


class ROIPanel(QGroupBox):
    """
    Panel for interactive ROI (Region of Interest) box selection.
    """
    roi_toggled = pyqtSignal(bool)
    apply_roi = pyqtSignal()
    reset_roi = pyqtSignal()

    def __init__(self, title: str = "📦 ROI Selection"):
        super().__init__()
        self.custom_title = title
        self._enabled = False
        self._bounds = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        layout.addWidget(title_lbl)

        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable ROI Selection")
        self.enable_checkbox.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.enable_checkbox)

        # Bounds display label
        self.bounds_label = QLabel("Bounds: Not selected")
        self.bounds_label.setWordWrap(True)
        layout.addWidget(self.bounds_label)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("✓ Apply ROI")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_roi.emit)
        btn_layout.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("↩ Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _on_toggle(self, state):
        self._enabled = (state == Qt.Checked)
        self.roi_toggled.emit(self._enabled)
        if not self._enabled:
            self._bounds = None
            self.bounds_label.setText("Bounds: Not selected")
            self.apply_btn.setEnabled(False)

    def _on_reset(self):
        self.enable_checkbox.setChecked(False)
        self._bounds = None
        self.bounds_label.setText("Bounds: Not selected")
        self.apply_btn.setEnabled(False)
        self.reset_roi.emit()

    def update_bounds(self, bounds: tuple):
        """Called by Visualizer when box widget is moved."""
        self._bounds = bounds
        if bounds:
            self.bounds_label.setText(
                f"X: {bounds[0]:.1f} - {bounds[1]:.1f}\n"
                f"Y: {bounds[2]:.1f} - {bounds[3]:.1f}\n"
                f"Z: {bounds[4]:.1f} - {bounds[5]:.1f}"
            )
            self.apply_btn.setEnabled(True)
        else:
            self.bounds_label.setText("Bounds: Not selected")
            self.apply_btn.setEnabled(False)

    def get_bounds(self) -> tuple:
        return self._bounds

    def is_enabled(self) -> bool:
        return self._enabled
