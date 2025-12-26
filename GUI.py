from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QGroupBox, QSlider, QComboBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# ==========================================
# Reusable UI Components
# ==========================================

class Separator(QFrame):
    """Simple horizontal line separator"""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class InfoPanel(QGroupBox):
    """Panel to display sample metadata and statistics"""

    def __init__(self, title="Sample Information"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.lbl_type = QLabel("Type: No data loaded")
        self.lbl_dim = QLabel("Grid: N/A")
        self.lbl_spacing = QLabel("Voxel Size: N/A")
        self.lbl_meta = QLabel("Sample Data: N/A")

        for lbl in [self.lbl_type, self.lbl_dim, self.lbl_spacing, self.lbl_meta]:
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        self.setLayout(layout)

    def update_info(self, type_str, dim, spacing, meta_dict):
        self.lbl_type.setText(f"Type: {type_str}")
        self.lbl_dim.setText(f"Grid: {dim}")
        self.lbl_spacing.setText(f"Voxel Size: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")

        meta_str = "\n".join([f"{k}: {v}" for k, v in meta_dict.items() if k != 'Type'][:3])
        self.lbl_meta.setText(f"Sample Data:\n{meta_str}" if meta_str else "Sample Data: None")


class AnalysisModePanel(QGroupBox):
    """Panel for switching between different 3D views"""

    # Signals to notify the visualizer to change view
    mode_changed = pyqtSignal(str)  # "volume", "slices", "iso", "clear", "reset"

    def __init__(self, title="Analysis Modes"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        btn_vol = QPushButton("📊 Volume Rendering")
        btn_vol.clicked.connect(lambda: self.mode_changed.emit("volume"))

        btn_slice = QPushButton("🔳 Orthogonal Slices")
        btn_slice.clicked.connect(lambda: self.mode_changed.emit("slices"))

        btn_iso = QPushButton("🏔️ Isosurface (Solid/Pore)")
        btn_iso.clicked.connect(lambda: self.mode_changed.emit("iso"))

        btn_clear = QPushButton("🗑️ Clear View")
        btn_clear.clicked.connect(lambda: self.mode_changed.emit("clear"))

        btn_reset = QPushButton("🎥 Reset Camera")
        btn_reset.clicked.connect(lambda: self.mode_changed.emit("reset"))

        for btn in [btn_vol, btn_slice, btn_iso]:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        layout.addWidget(Separator())

        for btn in [btn_clear, btn_reset]:
            btn.setMinimumHeight(35)
            layout.addWidget(btn)

        self.setLayout(layout)


class RenderingParamsPanel(QGroupBox):
    """Panel for adjusting rendering parameters (threshold, colormap)"""

    threshold_changed = pyqtSignal(int)
    colormap_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(str)

    def __init__(self, title="Rendering Parameters"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Threshold
        layout.addWidget(QLabel("Iso-Threshold (Intensity):"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-1000, 2000)
        self.slider.setValue(300)
        self.slider.setTickInterval(200)
        self.slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self.slider)

        self.lbl_val = QLabel("Value: 300 Intensity")
        self.lbl_val.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_val)

        # Colormap
        layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['bone', 'viridis', 'plasma', 'gray', 'coolwarm', 'jet'])
        self.combo_cmap.currentTextChanged.connect(self.colormap_changed.emit)
        layout.addWidget(self.combo_cmap)

        # Opacity
        layout.addWidget(QLabel("Opacity Preset:"))
        self.combo_op = QComboBox()
        self.combo_op.addItems(['sigmoid', 'sigmoid_10', 'linear', 'linear_r', 'geom', 'geom_r'])
        self.combo_op.currentTextChanged.connect(self.opacity_changed.emit)
        layout.addWidget(self.combo_op)

        self.setLayout(layout)

    def _on_slider_change(self, val):
        self.lbl_val.setText(f"Value: {val} Intensity")
        self.threshold_changed.emit(val)

    def get_current_threshold(self):
        return self.slider.value()

    def get_current_colormap(self):
        return self.combo_cmap.currentText()

    def get_current_opacity(self):
        return self.combo_op.currentText()


class StructureProcessingPanel(QGroupBox):
    """
    Panel specific to the Application Logic (Controller).
    Decoupled from the Visualizer logic.
    """

    # Signals for Controller actions
    load_clicked = pyqtSignal()
    fast_load_clicked = pyqtSignal()
    dummy_clicked = pyqtSignal()
    extract_pores_clicked = pyqtSignal()
    pnm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self, title="Structure Processing"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Loaders
        btn_load = QPushButton("📁 Load Sample Scan")
        btn_load.clicked.connect(self.load_clicked.emit)

        btn_fast = QPushButton("⚡ Fast Load (Low-Res)")
        btn_fast.clicked.connect(self.fast_load_clicked.emit)

        btn_dummy = QPushButton("🧪 Load Synthetic Sample")
        btn_dummy.clicked.connect(self.dummy_clicked.emit)

        for btn in [btn_load, btn_fast, btn_dummy]:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        layout.addWidget(Separator())

        # Processors
        btn_pores = QPushButton("🔬 Extract Void Space")
        btn_pores.clicked.connect(self.extract_pores_clicked.emit)

        btn_pnm = QPushButton("⚪ Pore Network Model (PNM)")
        btn_pnm.clicked.connect(self.pnm_clicked.emit)

        for btn in [btn_pores, btn_pnm]:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        layout.addWidget(Separator())

        # Reset
        btn_reset = QPushButton("↩️ Reset to Raw Data")
        btn_reset.setMinimumHeight(35)
        btn_reset.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(btn_reset)

        self.setLayout(layout)