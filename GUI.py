from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QGroupBox, QSlider, QComboBox, QFrame, QHBoxLayout)
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

    def update_info(self, type_str, dim, spacing, metadata):
        """Update labels with new data"""
        self.lbl_type.setText(f"Type: {type_str}")
        self.lbl_dim.setText(f"Grid: {dim}")
        self.lbl_spacing.setText(f"Voxel Size: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")

        meta_str = "\n".join([f"{k}: {v}" for k, v in metadata.items() if k != 'Type'][:3])
        self.lbl_meta.setText(f"Sample Data:\n{meta_str}" if meta_str else "Sample Data: None")


class StructureProcessingPanel(QGroupBox):
    """
    Panel for Workflow Actions (Load, Process, Model).
    """
    # Signals to communicate with Controller
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

        btn_reset = QPushButton("↩️ Reset to Raw Data")
        btn_reset.clicked.connect(self.reset_clicked.emit)
        btn_reset.setMinimumHeight(35)
        layout.addWidget(btn_reset)

        self.setLayout(layout)


class VisualizationModePanel(QGroupBox):
    """
    Panel for selecting the Visualization Mode (Volume, Slices, Iso).
    Moved from Visualizers.py.
    """
    volume_clicked = pyqtSignal()
    slices_clicked = pyqtSignal()
    iso_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    reset_camera_clicked = pyqtSignal()

    def __init__(self, title="Analysis Modes"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        self.btn_volume = QPushButton("📊 Volume Rendering")
        self.btn_volume.setMinimumHeight(40)
        self.btn_volume.clicked.connect(self.volume_clicked.emit)
        layout.addWidget(self.btn_volume)

        self.btn_slices = QPushButton("🔳 Orthogonal Slices")
        self.btn_slices.setMinimumHeight(40)
        self.btn_slices.clicked.connect(self.slices_clicked.emit)
        layout.addWidget(self.btn_slices)

        self.btn_isosurface = QPushButton("🏔️ Isosurface (Solid/Pore)")
        self.btn_isosurface.setMinimumHeight(40)
        self.btn_isosurface.clicked.connect(self.iso_clicked.emit)
        layout.addWidget(self.btn_isosurface)

        layout.addWidget(Separator())

        self.btn_clear = QPushButton("🗑️ Clear View")
        self.btn_clear.setMinimumHeight(35)
        self.btn_clear.clicked.connect(self.clear_clicked.emit)
        layout.addWidget(self.btn_clear)

        self.btn_reset_camera = QPushButton("🎥 Reset Camera")
        self.btn_reset_camera.setMinimumHeight(35)
        self.btn_reset_camera.clicked.connect(self.reset_camera_clicked.emit)
        layout.addWidget(self.btn_reset_camera)

        self.setLayout(layout)


class RenderingParametersPanel(QGroupBox):
    """
    Panel for controlling fine-grained rendering parameters.
    Handles visibility logic for widgets based on the current mode.
    Moved from Visualizers.py.
    """
    # Signals carrying values
    threshold_changed = pyqtSignal(int)
    coloring_mode_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    solid_color_changed = pyqtSignal(str)
    light_angle_changed = pyqtSignal(int)
    render_style_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(str)

    def __init__(self, title="Rendering Parameters"):
        super().__init__(title)
        self.active_mode = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # 1. Threshold (Iso only)
        self.lbl_threshold = QLabel("Iso-Threshold (Intensity):")
        layout.addWidget(self.lbl_threshold)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(-1000)
        self.threshold_slider.setMaximum(2000)
        self.threshold_slider.setValue(300)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(200)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("Value: 300 Intensity")
        self.threshold_value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.threshold_value_label)

        # 2. Coloring Mode Selector (Iso only)
        self.lbl_coloring_mode = QLabel("Isosurface Coloring Mode:")
        layout.addWidget(self.lbl_coloring_mode)

        self.coloring_mode_combo = QComboBox()
        self.coloring_mode_combo.addItems([
            'Solid Color',
            'Depth (Z-Axis)',
            'Radial (Center Dist)'
        ])
        self.coloring_mode_combo.setToolTip("Select how the isosurface is colored.")
        self.coloring_mode_combo.setCurrentIndex(0)
        self.coloring_mode_combo.currentTextChanged.connect(self.coloring_mode_changed.emit)
        # Also trigger visibility update internally when mode changes
        self.coloring_mode_combo.currentTextChanged.connect(self._update_visibility)
        layout.addWidget(self.coloring_mode_combo)

        # 3. Colormap selector (Volume, Slices, or Iso-Scalar)
        self.lbl_colormap = QLabel("Colormap:")
        layout.addWidget(self.lbl_colormap)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['bone', 'viridis', 'plasma', 'gray', 'coolwarm', 'jet', 'magma'])
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        layout.addWidget(self.colormap_combo)

        # 4. Solid Color Selector (Iso-Solid only)
        self.lbl_solid_color = QLabel("Solid Color:")
        layout.addWidget(self.lbl_solid_color)

        self.solid_color_combo = QComboBox()
        self.solid_color_combo.addItems(['ivory', 'red', 'gold', 'lightgray', 'mediumseagreen', 'dodgerblue', 'wheat'])
        self.solid_color_combo.currentTextChanged.connect(self.solid_color_changed.emit)
        layout.addWidget(self.solid_color_combo)

        # 5. Light Azimuth (Position) Slider (Iso only)
        self.lbl_light_angle = QLabel("Light Source Angle (0-360°):")
        layout.addWidget(self.lbl_light_angle)

        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(0, 360)
        self.light_azimuth_slider.setValue(45)
        self.light_azimuth_slider.setTickInterval(45)
        self.light_azimuth_slider.valueChanged.connect(self.light_angle_changed.emit)
        layout.addWidget(self.light_azimuth_slider)

        # 6. Render Style Selector
        self.lbl_render_style = QLabel("Render Style (显示风格):")
        layout.addWidget(self.lbl_render_style)

        self.render_style_combo = QComboBox()
        self.render_style_combo.addItems(['Surface', 'Wireframe', 'Wireframe + Surface'])
        self.render_style_combo.setCurrentIndex(0)
        self.render_style_combo.currentTextChanged.connect(self.render_style_changed.emit)
        layout.addWidget(self.render_style_combo)

        # 7. Opacity preset selector (Volume only)
        self.lbl_opacity = QLabel("Opacity Preset:")
        layout.addWidget(self.lbl_opacity)

        self.opacity_combo = QComboBox()
        self.opacity_combo.addItems(['sigmoid', 'sigmoid_10', 'linear', 'linear_r', 'geom', 'geom_r'])
        self.opacity_combo.currentTextChanged.connect(self.opacity_changed.emit)
        layout.addWidget(self.opacity_combo)

        self.setLayout(layout)

        # Initial state
        self._update_visibility()

    def set_mode(self, mode):
        """Set the current visualization mode ('volume', 'slices', 'iso', None)"""
        self.active_mode = mode
        self._update_visibility()

    def get_current_values(self):
        """Helper to get all current values at once"""
        return {
            'threshold': self.threshold_slider.value(),
            'coloring_mode': self.coloring_mode_combo.currentText(),
            'colormap': self.colormap_combo.currentText(),
            'solid_color': self.solid_color_combo.currentText(),
            'light_angle': self.light_azimuth_slider.value(),
            'render_style': self.render_style_combo.currentText(),
            'opacity': self.opacity_combo.currentText()
        }

    def _on_threshold_change(self, value):
        self.threshold_value_label.setText(f"Value: {value} Intensity")
        self.threshold_changed.emit(value)

    def _update_visibility(self):
        """Internal logic to show/hide widgets based on mode"""
        mode = self.active_mode

        # Helper to toggle visibility
        def set_visible(widgets, visible):
            for w in widgets:
                w.setVisible(visible)

        show_threshold = (mode == 'iso')
        show_coloring_mode = (mode == 'iso')
        show_light_angle = (mode == 'iso')
        show_render_style = (mode == 'iso')
        show_opacity = (mode == 'volume')

        show_colormap = False
        show_solid_color = False

        if mode == 'volume':
            show_colormap = True
        elif mode == 'slices':
            show_colormap = True
        elif mode == 'iso':
            if self.coloring_mode_combo.currentText() == 'Solid Color':
                show_solid_color = True
            else:
                show_colormap = True

        set_visible([self.lbl_threshold, self.threshold_slider, self.threshold_value_label], show_threshold)
        set_visible([self.lbl_coloring_mode, self.coloring_mode_combo], show_coloring_mode)
        set_visible([self.lbl_light_angle, self.light_azimuth_slider], show_light_angle)
        set_visible([self.lbl_render_style, self.render_style_combo], show_render_style)
        set_visible([self.lbl_opacity, self.opacity_combo], show_opacity)
        set_visible([self.lbl_colormap, self.colormap_combo], show_colormap)
        set_visible([self.lbl_solid_color, self.solid_color_combo], show_solid_color)