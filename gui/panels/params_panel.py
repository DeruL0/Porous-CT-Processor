"""
Rendering Parameters Panel for controlling visualization settings.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QSlider, QComboBox, QSpinBox, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from typing import Dict, Any
import numpy as np
import pyqtgraph as pg

from gui.styles import PANEL_TITLE_STYLE
from config import (
    DEFAULT_COLORMAPS, 
    SOLID_COLORS, 
    OPACITY_PRESETS, 
    RENDER_STYLES, 
    COLORING_MODES
)


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
    apply_clim_clip = pyqtSignal(list)  # Emits [min, max] for permanent clipping
    invert_volume = pyqtSignal()        # Emits signal to invert volume values
    invert_volume = pyqtSignal()        # Emits signal to invert volume values

    def __init__(self, title: str = "🎨 Rendering Parameters"):
        super().__init__()
        self.custom_title = title
        self.active_mode = None
        self._is_updating_programmatically = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        layout.addWidget(title_lbl)

        # 1. Threshold (Iso)
        self.lbl_threshold, self.threshold_slider, self.threshold_value_label = self._create_slider_group(
            layout, "Iso-Threshold (Intensity):", -1000, 2000, 300, self._on_threshold_change
        )

        # 2. Coloring Mode (Iso)
        self.lbl_coloring_mode = QLabel("Isosurface Coloring Mode:")
        layout.addWidget(self.lbl_coloring_mode)
        self.coloring_mode_combo = self._create_combo(layout,
                                                      COLORING_MODES,
                                                      [self.coloring_mode_changed.emit, self._update_visibility]
                                                      )

        # 3. Slice Sliders (Slices)
        self.lbl_slice_x, self.slider_slice_x, _ = self._create_slider_group(
            layout, "Slice X Position:", 0, 100, 50, lambda v: self.slice_position_changed.emit())
        self.lbl_slice_y, self.slider_slice_y, _ = self._create_slider_group(
            layout, "Slice Y Position:", 0, 100, 50, lambda v: self.slice_position_changed.emit())
        self.lbl_slice_z, self.slider_slice_z, _ = self._create_slider_group(
            layout, "Slice Z Position:", 0, 100, 50, lambda v: self.slice_position_changed.emit())

        # 4. Colormap Selection
        self.lbl_colormap = QLabel("Colormap:")
        layout.addWidget(self.lbl_colormap)
        self.colormap_combo = self._create_combo(layout,
                                                 DEFAULT_COLORMAPS,
                                                 [self.colormap_changed.emit]
                                                 )

        # 5. Colormap Range (CLim)
        self.lbl_clim = QLabel("Colormap Range (Min/Max):")
        layout.addWidget(self.lbl_clim)

        # Histogram Widget
        self.hist_widget = pg.PlotWidget()
        self.hist_widget.setBackground('w')
        self.hist_widget.setFixedHeight(120)
        self.hist_widget.setMouseEnabled(x=False, y=False)
        self.hist_widget.hideAxis('left')
        self.hist_widget.hideAxis('bottom')
        
        # Region Item (draggable area)
        self.hist_region = pg.LinearRegionItem()
        self.hist_region.setZValue(10)
        self.hist_region.sigRegionChanged.connect(self._on_histogram_region_changed)
        self.hist_widget.addItem(self.hist_region)
        
        # Mapping Curve (Ramp)
        self.mapping_curve = pg.PlotCurveItem(pen=pg.mkPen('r', width=3))
        self.hist_widget.addItem(self.mapping_curve)
        
        self.hist_data_bounds = (0, 100)
        self.hist_peak = 1.0
        self.hist_data_bounds = (0, 100)
        self.hist_peak = 1.0
        
        layout.addWidget(self.hist_widget)
        


        # Min row

        # Min row
        min_container = QWidget()
        min_layout = QHBoxLayout(min_container)
        min_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_clim_min = QLabel("Min:")
        min_layout.addWidget(self.lbl_clim_min)
        self.slider_clim_min = QSlider(Qt.Horizontal)
        self.slider_clim_min.setRange(0, 100)
        self.slider_clim_min.setValue(0)
        self.slider_clim_min.valueChanged.connect(self._on_clim_slider_change)
        min_layout.addWidget(self.slider_clim_min, stretch=1)
        self.spinbox_clim_min = QSpinBox()
        self.spinbox_clim_min.setRange(0, 100)
        self.spinbox_clim_min.setValue(0)
        self.spinbox_clim_min.valueChanged.connect(self._on_clim_spinbox_change)
        min_layout.addWidget(self.spinbox_clim_min)
        layout.addWidget(min_container)

        # Max row
        max_container = QWidget()
        max_layout = QHBoxLayout(max_container)
        max_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_clim_max = QLabel("Max:")
        max_layout.addWidget(self.lbl_clim_max)
        self.slider_clim_max = QSlider(Qt.Horizontal)
        self.slider_clim_max.setRange(0, 100)
        self.slider_clim_max.setValue(100)
        self.slider_clim_max.valueChanged.connect(self._on_clim_slider_change)
        max_layout.addWidget(self.slider_clim_max, stretch=1)
        self.spinbox_clim_max = QSpinBox()
        self.spinbox_clim_max.setRange(0, 100)
        self.spinbox_clim_max.setValue(100)
        self.spinbox_clim_max.valueChanged.connect(self._on_clim_spinbox_change)
        max_layout.addWidget(self.spinbox_clim_max)
        layout.addWidget(max_container)

        # Apply Clip button
        self.btn_apply_clim_clip = QPushButton("✂️ Apply Range Clip")
        self.btn_apply_clim_clip.setToolTip("Permanently clip data to current min/max range")
        self.btn_apply_clim_clip.clicked.connect(self._on_apply_clim_clip)
        layout.addWidget(self.btn_apply_clim_clip)

        # Invert Volume button
        self.btn_invert_volume = QPushButton("🔄 Invert Volume")
        self.btn_invert_volume.setToolTip("Invert volume values (extract pore surfaces instead of object surfaces)")
        self.btn_invert_volume.clicked.connect(self.invert_volume.emit)
        layout.addWidget(self.btn_invert_volume)

        # 6. Solid Color
        self.lbl_solid_color = QLabel("Solid Color:")
        layout.addWidget(self.lbl_solid_color)
        self.solid_color_combo = self._create_combo(layout, SOLID_COLORS, [self.solid_color_changed.emit])

        # 7. Light Angle
        self.lbl_light_angle, self.light_azimuth_slider, _ = self._create_slider_group(
            layout, "Light Source Angle (0-360°):", 0, 360, 45, self.light_angle_changed.emit
        )

        # 8. Render Style
        self.lbl_render_style = QLabel("Render Style:")
        layout.addWidget(self.lbl_render_style)
        self.render_style_combo = self._create_combo(layout, RENDER_STYLES, [self.render_style_changed.emit])

        # 9. Opacity
        self.lbl_opacity = QLabel("Opacity Preset:")
        layout.addWidget(self.lbl_opacity)
        self.opacity_combo = self._create_combo(layout, OPACITY_PRESETS, [self.opacity_changed.emit])

        self.setLayout(layout)
        self._update_visibility()

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
        self.spinbox_clim_min.setRange(int(min_val), int(max_val))
        self.spinbox_clim_max.setRange(int(min_val), int(max_val))
        self.slider_clim_min.setValue(int(min_val))
        self.slider_clim_max.setValue(int(max_val))
        self.spinbox_clim_min.setValue(int(min_val))
        self.spinbox_clim_max.setValue(int(max_val))
        self.spinbox_clim_max.setValue(int(max_val))
        
        # Update histogram region if visible
        if hasattr(self, 'hist_region'):
            self.hist_region.setBounds([min_val, max_val])
            self.hist_region.setRegion([min_val, max_val])
            
        self.block_signals(False)

    def set_histogram_data(self, hist: np.ndarray, bins: np.ndarray):
        """Update histogram plot with new data."""
        self.hist_widget.clear()
        self.hist_widget.addItem(self.hist_region)
        self.hist_widget.addItem(self.mapping_curve)
        
        # Store bounds for mapping curve (ensure float)
        mn, mx = float(bins[0]), float(bins[-1])
        self.hist_data_bounds = (mn, mx)
        if hist.size > 0:
            self.hist_peak = float(hist.max())
        
        # Use stepMode=True which expects len(x) == len(y) + 1
        # bins has length N+1, hist has length N
        curve = pg.PlotCurveItem(bins, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 100), pen='k')
        self.hist_widget.addItem(curve)
        
        # Set range (use float values to avoid pyqtgraph casting overflow warnings)
        self.hist_widget.setXRange(mn, mx, padding=0)
        self.hist_region.setBounds([mn, mx])
        
        # Update mapping curve
        self._update_mapping_curve()

    def _update_mapping_curve(self):
        """Update the visual ramp curve based on current region."""
        if not hasattr(self, 'mapping_curve'):
            return
            
        mn, mx = self.hist_region.getRegion()
        d_min, d_max = self.hist_data_bounds
        peak = self.hist_peak
        
        # Simple linear ramp from mn to mx
        x_vals = [d_min, mn, mx, d_max]
        y_vals = [0, 0, peak, peak]
        
        self.mapping_curve.setData(x_vals, y_vals)
    


    def block_signals(self, block: bool):
        self.slider_slice_x.blockSignals(block)
        self.slider_slice_y.blockSignals(block)
        self.slider_slice_z.blockSignals(block)
        self.slider_clim_min.blockSignals(block)
        self.slider_clim_max.blockSignals(block)
        self.spinbox_clim_min.blockSignals(block)
        self.spinbox_clim_max.blockSignals(block)

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

    def _on_clim_slider_change(self, value):
        self.spinbox_clim_min.blockSignals(True)
        self.spinbox_clim_max.blockSignals(True)
        self.spinbox_clim_min.setValue(self.slider_clim_min.value())
        self.spinbox_clim_max.setValue(self.slider_clim_max.value())
        self.spinbox_clim_min.blockSignals(False)
        self.spinbox_clim_max.blockSignals(False)
        if self.slider_clim_min.value() > self.slider_clim_max.value():
            self.slider_clim_max.setValue(self.slider_clim_min.value())
            
        # Sync histogram region
        if not self._is_updating_programmatically:
            self.hist_region.setRegion([self.slider_clim_min.value(), self.slider_clim_max.value()])
            self._update_mapping_curve()
            
        self.clim_changed.emit()

    def _on_clim_spinbox_change(self, value):
        self.slider_clim_min.blockSignals(True)
        self.slider_clim_max.blockSignals(True)
        self.slider_clim_min.setValue(self.spinbox_clim_min.value())
        self.slider_clim_max.setValue(self.spinbox_clim_max.value())
        self.slider_clim_min.blockSignals(False)
        self.slider_clim_max.blockSignals(False)
        if self.spinbox_clim_min.value() > self.spinbox_clim_max.value():
            self.spinbox_clim_max.setValue(self.spinbox_clim_min.value())
            
        # Sync histogram region
        if not self._is_updating_programmatically:
            self.hist_region.setRegion([self.spinbox_clim_min.value(), self.spinbox_clim_max.value()])
            self._update_mapping_curve()
            
        self.clim_changed.emit()

    def _on_histogram_region_changed(self):
        """Sync histogram region to sliders/spinboxes."""
        self._is_updating_programmatically = True
        min_val, max_val = self.hist_region.getRegion()
        
        self.slider_clim_min.setValue(int(min_val))
        self.slider_clim_max.setValue(int(max_val))
        self.spinbox_clim_min.setValue(int(min_val))
        self.spinbox_clim_max.setValue(int(max_val))
        
        self._update_mapping_curve()
        
        self._is_updating_programmatically = False
        self.clim_changed.emit()

    def _on_apply_clim_clip(self):
        """Emit signal to permanently clip data to current clim range."""
        clim = [self.slider_clim_min.value(), self.slider_clim_max.value()]
        self.apply_clim_clip.emit(clim)

    def _update_visibility(self):
        mode = self.active_mode
        is_iso = (mode == 'iso')
        is_vol = (mode == 'volume')
        is_slice = (mode == 'slices')
        is_mesh = (mode == 'mesh')

        show_solid_col = is_iso and (self.coloring_mode_combo.currentText() == 'Solid Color')
        show_cmap = is_vol or is_slice or is_mesh or (is_iso and not show_solid_col)
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

        visible([self.lbl_clim, self.hist_widget, 
                 self.lbl_clim_min, self.slider_clim_min, self.spinbox_clim_min,
                 self.lbl_clim_max, self.slider_clim_max, self.spinbox_clim_max,
                 self.btn_apply_clim_clip, self.btn_invert_volume], show_clim)

        visible([self.lbl_slice_x, self.slider_slice_x], is_slice)
        visible([self.lbl_slice_y, self.slider_slice_y], is_slice)
        visible([self.lbl_slice_z, self.slider_slice_z], is_slice)
