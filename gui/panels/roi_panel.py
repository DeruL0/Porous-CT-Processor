"""
ROI Panel for interactive Region of Interest selection.
Supports Box, Cylinder, and Sphere shapes - all use box widget for easy manipulation.
Cylinder and Sphere are inscribed within the box bounds.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QPushButton, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from gui.styles import PANEL_TITLE_STYLE


class ROIPanel(QGroupBox):
    """
    Panel for interactive ROI (Region of Interest) selection.
    All shapes use box widget - Cylinder and Sphere are inscribed within bounds.
    """
    roi_toggled = pyqtSignal(bool)
    apply_roi = pyqtSignal()
    reset_roi = pyqtSignal()
    shape_changed = pyqtSignal(str)  # Emits: 'box', 'cylinder', 'sphere'

    def __init__(self, title: str = "📦 ROI Selection"):
        super().__init__()
        self.custom_title = title
        self._enabled = False
        self._bounds = None
        self._shape = 'box'
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        layout.addWidget(title_lbl)

        # Shape selector
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Shape:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Box", "Cylinder", "Sphere"])
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        shape_layout.addWidget(self.shape_combo)
        layout.addLayout(shape_layout)
        
        # Shape description
        self.shape_desc = QLabel("Extract rectangular region")
        self.shape_desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.shape_desc)

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

    def _on_shape_changed(self, text: str):
        shape = text.lower()
        self._shape = shape
        
        # Update description based on shape
        descriptions = {
            'box': "Extract rectangular region",
            'cylinder': "Extract cylinder inscribed in box (YZ plane)",
            'sphere': "Extract sphere inscribed in box"
        }
        self.shape_desc.setText(descriptions.get(shape, ""))
        
        # Keep widget enabled - just emit shape change
        self._bounds = None
        self.bounds_label.setText("Bounds: Not selected")
        self.apply_btn.setEnabled(False)
        
        self.shape_changed.emit(shape)

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
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
            
            shape_info = ""
            if self._shape == 'sphere':
                # Ellipsoid: show all 3 radii
                rx, ry, rz = size_x / 2, size_y / 2, size_z / 2
                shape_info = f"\nEllipsoid Rx:{rx:.1f} Ry:{ry:.1f} Rz:{rz:.1f}"
            elif self._shape == 'cylinder':
                # Elliptical cylinder: Y/Z radii, X height
                ry, rz = size_y / 2, size_z / 2
                shape_info = f"\nCylinder Ry:{ry:.1f} Rz:{rz:.1f} H:{size_x:.1f}"
            
            self.bounds_label.setText(
                f"X: {bounds[0]:.1f} - {bounds[1]:.1f}\n"
                f"Y: {bounds[2]:.1f} - {bounds[3]:.1f}\n"
                f"Z: {bounds[4]:.1f} - {bounds[5]:.1f}{shape_info}"
            )
            self.apply_btn.setEnabled(True)
        else:
            self.bounds_label.setText("Bounds: Not selected")
            self.apply_btn.setEnabled(False)

    def get_shape(self) -> str:
        """Get current shape selection."""
        return self._shape

    def get_bounds(self) -> tuple:
        """Get bounds for all shapes (box widget bounds)."""
        return self._bounds

    def is_enabled(self) -> bool:
        return self._enabled
