from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QGroupBox, QSlider, QComboBox, QFrame, QTableWidget,
                             QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Dict, Any, List
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==========================================
# Statistics Panel (Merged)
# ==========================================

class StatisticsPanel(QWidget):
    """
    Panel to display advanced statistics and visualizations for PNM analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(400)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Statistics & Analysis")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Matplotlib Figure for Histogram
        self.figure = Figure(figsize=(5, 3), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        layout.addWidget(self.canvas)
        
        # Statistics Table
        self.stats_table = QTableWidget()
        self.stats_table.setMaximumHeight(120)
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stats_table)
        
        layout.addStretch()
        
    def update_statistics(self, metadata):
        """Update the panel with new statistics from VolumeData metadata."""
        # Clear previous
        self.figure.clear()
        self.stats_table.setRowCount(0)
        
        # Extract metrics
        size_dist = metadata.get("PoreSizeDistribution", {})
        largest_pore = metadata.get("LargestPoreRatio", "N/A")
        throat_stats = metadata.get("ThroatStats", {})
        pore_count = metadata.get("PoreCount", 0)
        connection_count = metadata.get("ConnectionCount", 0)
        
        # Plot histogram
        if size_dist.get("bins") and size_dist.get("counts"):
            ax = self.figure.add_subplot(111)
            bins = np.array(size_dist["bins"])
            counts = size_dist["counts"]
            
            # Bar plot
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, counts, width=np.diff(bins), 
                   alpha=0.7, color='steelblue', edgecolor='black')
            
            ax.set_xlabel('Pore Radius (mm)')
            ax.set_ylabel('Count')
            ax.set_title('Pore Size Distribution')
            ax.grid(True, alpha=0.3)
            self.figure.tight_layout()
            
        self.canvas.draw()
        
        # Populate stats table
        stats_data = [
            ("Total Pores", str(pore_count)),
            ("Connections", str(connection_count)),
            ("Largest Pore %", str(largest_pore)),
        ]
        
        if throat_stats:
            stats_data.extend([
                ("Throat Min (mm)", f"{throat_stats.get('min', 0):.3f}"),
                ("Throat Max (mm)", f"{throat_stats.get('max', 0):.3f}"),
                ("Throat Mean (mm)", f"{throat_stats.get('mean', 0):.3f}"),
            ])
        
        self.stats_table.setRowCount(len(stats_data))
        for row, (metric, value) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))
        
    def clear(self):
        """Clear all statistics."""
        self.figure.clear()
        self.canvas.draw()
        self.stats_table.setRowCount(0)


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

    def __init__(self, title: str = "Sample Information"):
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

    def update_info(self, type_str: str, dim: tuple, spacing: tuple, metadata: dict):
        """Update labels with new data"""
        self.lbl_type.setText(f"Type: {type_str}")

        if dim == (0, 0, 0) and 'MeshPoints' in metadata:
            self.lbl_dim.setText(f"Mesh Points: {metadata['MeshPoints']}")
        else:
            self.lbl_dim.setText(f"Grid: {dim}")

        self.lbl_spacing.setText(f"Voxel Size: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")

        # Intelligent Metadata Display
        # Prioritize Quantitative Analysis Keys
        priority_keys = ['Porosity', 'PoreCount', 'ConnectionCount', 'SampleID', 'MeshPoints']
        display_items = []

        # 1. Add priority items first
        for k in priority_keys:
            if k in metadata:
                display_items.append(f"{k}: {metadata[k]}")

        # 2. Add other items until we hit limit
        for k, v in metadata.items():
            if k not in priority_keys and k != 'Type':
                display_items.append(f"{k}: {v}")

        # Show top 5 items
        meta_str = "\n".join(display_items[:6])
        self.lbl_meta.setText(f"Sample Data:\n{meta_str}" if meta_str else "Sample Data: None")


class StructureProcessingPanel(QGroupBox):
    """
    Panel for Workflow Actions (Load, Process, Model, Export).
    """
    # Define Signals
    load_clicked = pyqtSignal()
    fast_load_clicked = pyqtSignal()
    dummy_clicked = pyqtSignal()
    extract_pores_clicked = pyqtSignal()
    pnm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    export_clicked = pyqtSignal()  # NEW SIGNAL

    def __init__(self, title: str = "Structure Processing"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Group 1: Loading
        self._add_button(layout, "📁 Load Sample Scan", self.load_clicked)
        self._add_button(layout, "⚡ Fast Load (Low-Res)", self.fast_load_clicked)
        self._add_button(layout, "🧪 Load Synthetic Sample", self.dummy_clicked)
        layout.addWidget(Separator())

        # Group 2: Processing
        self._add_button(layout, "🔬 Extract Void Space", self.extract_pores_clicked)
        self._add_button(layout, "⚪ Pore Network Model (PNM)", self.pnm_clicked)
        layout.addWidget(Separator())

        # Group 3: Export
        self._add_button(layout, "💾 Export to VTK", self.export_clicked, min_height=35)
        layout.addWidget(Separator())

        # Group 4: Reset
        self._add_button(layout, "↩️ Reset to Raw Data", self.reset_clicked, min_height=35)

        self.setLayout(layout)

    def _add_button(self, layout, text, signal, min_height=40):
        btn = QPushButton(text)
        btn.setMinimumHeight(min_height)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)


class VisualizationModePanel(QGroupBox):
    """
    Panel for selecting the Visualization Mode (Volume, Slices, Iso).
    """
    volume_clicked = pyqtSignal()
    slices_clicked = pyqtSignal()
    iso_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    reset_camera_clicked = pyqtSignal()

    def __init__(self, title: str = "Analysis Modes"):
        super().__init__(title)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
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
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)


class ClipPlanePanel(QGroupBox):
    """
    Panel for interactive clip plane controls.
    """
    clip_changed = pyqtSignal()  # Emits when any clip parameter changes
    clip_toggled = pyqtSignal(bool)  # Emits when clipping is enabled/disabled

    def __init__(self, title: str = "Clip Planes"):
        super().__init__(title)
        self._enabled = False
        self._init_ui()

    def _init_ui(self):
        from PyQt5.QtWidgets import QCheckBox, QHBoxLayout
        
        layout = QVBoxLayout()

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


class RenderingParametersPanel(QGroupBox):
    """
    Panel for controlling fine-grained rendering parameters.
    """
    # Signals
    threshold_changed = pyqtSignal(int)
    coloring_mode_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    solid_color_changed = pyqtSignal(str)
    light_angle_changed = pyqtSignal(int)
    render_style_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(str)
    slice_position_changed = pyqtSignal()
    clim_changed = pyqtSignal()

    def __init__(self, title: str = "Rendering Parameters"):
        super().__init__(title)
        self.active_mode = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # 1. Threshold (Iso)
        self.lbl_threshold, self.threshold_slider, self.threshold_value_label = self._create_slider_group(
            layout, "Iso-Threshold (Intensity):", -1000, 2000, 300, self._on_threshold_change
        )

        # 2. Coloring Mode (Iso)
        self.lbl_coloring_mode = QLabel("Isosurface Coloring Mode:")
        layout.addWidget(self.lbl_coloring_mode)
        self.coloring_mode_combo = self._create_combo(layout,
                                                      ['Solid Color', 'Depth (Z-Axis)', 'Radial (Center Dist)'],
                                                      [self.coloring_mode_changed.emit, self._update_visibility]
                                                      )

        # 3. Slice Sliders (Slices)
        self.lbl_slice_x, self.slider_slice_x, _ = self._create_slider_group(layout, "Slice X Position:", 0, 100, 50,
                                                                             lambda
                                                                                 v: self.slice_position_changed.emit())
        self.lbl_slice_y, self.slider_slice_y, _ = self._create_slider_group(layout, "Slice Y Position:", 0, 100, 50,
                                                                             lambda
                                                                                 v: self.slice_position_changed.emit())
        self.lbl_slice_z, self.slider_slice_z, _ = self._create_slider_group(layout, "Slice Z Position:", 0, 100, 50,
                                                                             lambda
                                                                                 v: self.slice_position_changed.emit())

        # 4. Colormap Selection
        self.lbl_colormap = QLabel("Colormap:")
        layout.addWidget(self.lbl_colormap)
        self.colormap_combo = self._create_combo(layout,
                                                 ['bone', 'viridis', 'plasma', 'gray', 'coolwarm', 'jet', 'magma'],
                                                 [self.colormap_changed.emit]
                                                 )

        # 5. Colormap Range (CLim)
        self.lbl_clim = QLabel("Colormap Range (Min/Max):")
        layout.addWidget(self.lbl_clim)

        self.lbl_clim_min, self.slider_clim_min, self.val_clim_min = self._create_slider_group(
            layout, "Min:", 0, 100, 0, self._on_clim_change
        )
        self.lbl_clim_max, self.slider_clim_max, self.val_clim_max = self._create_slider_group(
            layout, "Max:", 0, 100, 100, self._on_clim_change
        )

        # 6. Solid Color
        self.lbl_solid_color = QLabel("Solid Color:")
        layout.addWidget(self.lbl_solid_color)
        self.solid_color_combo = self._create_combo(layout,
                                                    ['ivory', 'red', 'gold', 'lightgray', 'mediumseagreen',
                                                     'dodgerblue', 'wheat'],
                                                    [self.solid_color_changed.emit]
                                                    )

        # 7. Light Angle
        self.lbl_light_angle, self.light_azimuth_slider, _ = self._create_slider_group(
            layout, "Light Source Angle (0-360°):", 0, 360, 45, self.light_angle_changed.emit
        )

        # 8. Render Style
        self.lbl_render_style = QLabel("Render Style:")
        layout.addWidget(self.lbl_render_style)
        self.render_style_combo = self._create_combo(layout,
                                                     ['Surface', 'Wireframe', 'Wireframe + Surface'],
                                                     [self.render_style_changed.emit]
                                                     )

        # 9. Opacity
        self.lbl_opacity = QLabel("Opacity Preset:")
        layout.addWidget(self.lbl_opacity)
        self.opacity_combo = self._create_combo(layout,
                                                ['sigmoid', 'sigmoid_10', 'linear', 'linear_r', 'geom', 'geom_r'],
                                                [self.opacity_changed.emit]
                                                )

        self.setLayout(layout)
        self._update_visibility()

    # --- UI Helpers ---

    def _create_slider_group(self, layout, label_text, min_val, max_val, default, callback):
        container = QWidget()
        h_layout = QVBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(2)

        label = QLabel(label_text)
        h_layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(callback)
        h_layout.addWidget(slider)

        val_label = QLabel(f"{default}")
        val_label.setAlignment(Qt.AlignRight)
        h_layout.addWidget(val_label)

        layout.addWidget(container)
        return label, slider, val_label

    def _create_combo(self, layout, items, callbacks):
        combo = QComboBox()
        combo.addItems(items)
        for cb in callbacks:
            combo.currentTextChanged.connect(cb)
        layout.addWidget(combo)
        return combo

    # --- Logic ---

    def set_mode(self, mode: str):
        self.active_mode = mode
        self._update_visibility()

    def set_slice_limits(self, x_max: int, y_max: int, z_max: int):
        self.slider_slice_x.setMaximum(x_max)
        self.slider_slice_y.setMaximum(y_max)
        self.slider_slice_z.setMaximum(z_max)

    def set_slice_defaults(self, x: int, y: int, z: int):
        self.block_signals(True)
        self.slider_slice_x.setValue(x)
        self.slider_slice_y.setValue(y)
        self.slider_slice_z.setValue(z)
        self.block_signals(False)

    def set_data_range(self, min_val: int, max_val: int):
        self.block_signals(True)
        self.slider_clim_min.setRange(int(min_val), int(max_val))
        self.slider_clim_max.setRange(int(min_val), int(max_val))
        self.slider_clim_min.setValue(int(min_val))
        self.slider_clim_max.setValue(int(max_val))
        self.val_clim_min.setText(f"{int(min_val)}")
        self.val_clim_max.setText(f"{int(max_val)}")
        self.block_signals(False)

    def block_signals(self, block: bool):
        self.slider_slice_x.blockSignals(block)
        self.slider_slice_y.blockSignals(block)
        self.slider_slice_z.blockSignals(block)
        self.slider_clim_min.blockSignals(block)
        self.slider_clim_max.blockSignals(block)

    def get_current_values(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold_slider.value(),
            'coloring_mode': self.coloring_mode_combo.currentText(),
            'colormap': self.colormap_combo.currentText(),
            'solid_color': self.solid_color_combo.currentText(),
            'light_angle': self.light_azimuth_slider.value(),
            'render_style': self.render_style_combo.currentText(),
            'opacity': self.opacity_combo.currentText(),
            'slice_x': self.slider_slice_x.value(),
            'slice_y': self.slider_slice_y.value(),
            'slice_z': self.slider_slice_z.value(),
            'clim': [self.slider_clim_min.value(), self.slider_clim_max.value()]
        }

    def _on_threshold_change(self, value):
        self.threshold_value_label.setText(f"Value: {value}")
        self.threshold_changed.emit(value)

    def _on_clim_change(self, value):
        self.val_clim_min.setText(f"{self.slider_clim_min.value()}")
        self.val_clim_max.setText(f"{self.slider_clim_max.value()}")
        if self.slider_clim_min.value() > self.slider_clim_max.value():
            self.slider_clim_max.setValue(self.slider_clim_min.value())
        self.clim_changed.emit()

    def _update_visibility(self):
        mode = self.active_mode
        is_iso = (mode == 'iso')
        is_vol = (mode == 'volume')
        is_slice = (mode == 'slices')
        # Mesh mode typically uses ISO logic for lighting, but no sliders needed usually.
        # But if 'mesh', we might want to hide most sliders or show scalar selection.
        # For now, treat 'mesh' as having minimal controls visible.

        show_solid_col = is_iso and (self.coloring_mode_combo.currentText() == 'Solid Color')
        show_cmap = is_vol or is_slice or (is_iso and not show_solid_col)
        show_clim = is_vol or is_slice

        def visible(widgets, status):
            for w in widgets:
                if w.parentWidget() != self:
                    w.parentWidget().setVisible(status)
                else:
                    w.setVisible(status)

        visible([self.lbl_threshold, self.threshold_slider, self.threshold_value_label], is_iso)
        visible([self.lbl_coloring_mode, self.coloring_mode_combo], is_iso)
        visible([self.lbl_light_angle, self.light_azimuth_slider], is_iso)
        visible([self.lbl_render_style, self.render_style_combo], is_iso)
        visible([self.lbl_solid_color, self.solid_color_combo], show_solid_col)

        visible([self.lbl_opacity, self.opacity_combo], is_vol)
        visible([self.lbl_colormap, self.colormap_combo], show_cmap)

        visible([self.lbl_clim, self.lbl_clim_min, self.slider_clim_min, self.val_clim_min,
                 self.lbl_clim_max, self.slider_clim_max, self.val_clim_max], show_clim)

        visible([self.lbl_slice_x, self.slider_slice_x], is_slice)
        visible([self.lbl_slice_y, self.slider_slice_y], is_slice)
        visible([self.lbl_slice_z, self.slider_slice_z], is_slice)