"""
Main application window combining PyQt5 UI with PyVista 3D rendering.
Delegates rendering logic to RenderEngine for separation of concerns.
"""

import math
from typing import Optional

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QFrame, QStatusBar, QScrollArea, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from core import BaseVisualizer, VolumeData
from gui.panels import (
    VisualizationModePanel,
    RenderingParametersPanel,
    InfoPanel,
    ClipPlanePanel,
    ROIPanel
)
from rendering import RenderEngine


# Resolve Metaclass conflict between PyQt5 and ABC
class _MainWindowMeta(type(QMainWindow), type(BaseVisualizer)):
    pass


class MainWindow(QMainWindow, BaseVisualizer, metaclass=_MainWindowMeta):
    """
    Main application window.
    Integrates PyQt5 UI controls with a PyVista 3D rendering canvas.
    Delegates rendering to RenderEngine for separation of concerns.
    """

    def __init__(self):
        super().__init__()
        self._data_manager = None

        self.setWindowTitle("Porous Media Analysis Suite")
        self.setGeometry(100, 100, 1400, 900)
        self._init_ui()

        # Initialize RenderEngine after UI is ready
        self.render_engine = RenderEngine(
            plotter=self.plotter,
            params_panel=self.params_panel,
            info_panel=self.info_panel,
            clip_panel=self.clip_panel,
            status_callback=self.update_status
        )

        # Debounce timer for expensive renders
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_delayed_render)

        self._setup_mouse_probe()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Controls)
        left_scroll = self._create_control_panel()
        self.main_splitter.addWidget(left_scroll)

        # Center Panel (3D Canvas)
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Structure Viewer"
        )
        self.main_splitter.addWidget(self.plotter.app_window)

        # Right Panel (Info)
        right_scroll = self._create_info_panel()
        self.main_splitter.addWidget(right_scroll)

        self.main_splitter.setSizes([350, 900, 350])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(2, False)

        main_layout.addWidget(self.main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")

    def _create_control_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        panel = QWidget()
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setSpacing(10)

        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())

        # Mode Panel
        self.mode_panel = VisualizationModePanel()
        self.mode_panel.volume_clicked.connect(lambda: self.render_volume(reset_view=True))
        self.mode_panel.slices_clicked.connect(lambda: self.render_slices(reset_view=True))
        self.mode_panel.iso_clicked.connect(self.render_isosurface_auto)
        self.mode_panel.clear_clicked.connect(self.clear_view)
        self.mode_panel.reset_camera_clicked.connect(self.reset_camera)
        layout.addWidget(self.mode_panel)

        # Parameters Panel
        self.params_panel = RenderingParametersPanel()
        for signal in [self.params_panel.solid_color_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.render_style_changed,
                       self.params_panel.threshold_changed,
                       self.params_panel.slice_position_changed]:
            signal.connect(self.trigger_render)

        self.params_panel.opacity_changed.connect(lambda: self.render_volume(reset_view=False))
        self.params_panel.clim_changed.connect(self._on_clim_changed_fast)
        self.params_panel.colormap_changed.connect(self._on_colormap_changed)
        self.params_panel.apply_clim_clip.connect(self._on_apply_clim_clip)
        self.params_panel.invert_volume.connect(self._on_invert_volume)
        layout.addWidget(self.params_panel)

        # Clip Panel
        self.clip_panel = ClipPlanePanel()
        self.clip_panel.clip_toggled.connect(self._on_clip_toggled)
        self.clip_update_timer = QTimer()
        self.clip_update_timer.setInterval(200)
        self.clip_update_timer.setSingleShot(True)
        self.clip_update_timer.timeout.connect(self._apply_clip_planes)
        self.clip_panel.clip_changed.connect(lambda: self.clip_update_timer.start())
        layout.addWidget(self.clip_panel)

        # ROI Panel
        self.roi_panel = ROIPanel()
        self.roi_panel.roi_toggled.connect(self._on_roi_toggled)
        self.roi_panel.apply_roi.connect(self._on_apply_roi)
        self.roi_panel.reset_roi.connect(self._on_reset_roi)
        layout.addWidget(self.roi_panel)

        layout.addStretch()
        scroll_area.setWidget(panel)
        return scroll_area

    def _create_info_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Information")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.right_splitter = QSplitter(Qt.Vertical)
        self.info_panel = InfoPanel()
        self.right_splitter.addWidget(self.info_panel)
        layout.addWidget(self.right_splitter)

        scroll_area.setWidget(panel)
        return scroll_area

    def add_custom_panel(self, panel: QWidget, index: int = 2, side: str = 'left'):
        if side == 'left' and hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)
        elif side == 'right' and hasattr(self, 'right_splitter'):
            self.right_splitter.addWidget(panel)
            self.right_splitter.setStretchFactor(0, 1)
            self.right_splitter.setStretchFactor(1, 1)

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def trigger_render(self):
        self.update_timer.start()

    def _perform_delayed_render(self):
        mode = self.render_engine.active_view_mode
        if mode == 'volume':
            self.render_volume(reset_view=False)
        elif mode == 'slices':
            self.render_slices(reset_view=False)
        elif mode == 'iso':
            self.render_isosurface_auto(reset_view=False)
        elif mode == 'mesh':
            self.render_mesh(reset_view=False)

    # ==========================================
    # BaseVisualizer Interface
    # ==========================================

    def set_data(self, data: VolumeData):
        """Set data and delegate to RenderEngine."""
        self.render_engine.set_data(data)
        d_type = data.metadata.get('Type', 'Unknown')

        if data.has_mesh:
            self.update_status(f"Loaded Mesh: {d_type}")
            self.render_mesh(reset_view=True)
            self.info_panel.update_info(d_type, (0, 0, 0), data.spacing, data.metadata)
        elif data.raw_data is not None:
            self.update_status(f"Loaded Volume: {d_type}")

            is_processed = ("Processed" in d_type)
            default_color = "red" if is_processed else "ivory"
            idx = self.params_panel.solid_color_combo.findText(default_color)
            if idx >= 0:
                self.params_panel.solid_color_combo.setCurrentIndex(idx)

            min_val = np.nanmin(data.raw_data)
            max_val = np.nanmax(data.raw_data)
            self.params_panel.set_data_range(min_val, max_val)

            dims = self.render_engine.grid.dimensions
            self.params_panel.set_slice_limits(dims[0] - 1, dims[1] - 1, dims[2] - 1)
            self.params_panel.set_slice_defaults(dims[0] // 2, dims[1] // 2, dims[2] // 2)

            self.render_volume(reset_view=True)
            self.info_panel.update_info(d_type, data.dimensions, data.spacing, data.metadata)

    @property
    def data(self) -> Optional[VolumeData]:
        return self.render_engine.data if hasattr(self, 'render_engine') else None

    @property
    def grid(self):
        return self.render_engine.grid if hasattr(self, 'render_engine') else None

    @property
    def mesh(self):
        return self.render_engine.mesh if hasattr(self, 'render_engine') else None

    @property
    def active_view_mode(self):
        return self.render_engine.active_view_mode if hasattr(self, 'render_engine') else None

    def show(self):
        super().show()
        self.plotter.app_window.show()

    def update_status(self, message: str):
        self.status_bar.showMessage(message)

    def set_data_manager(self, data_manager):
        self._data_manager = data_manager

    # ==========================================
    # Rendering Delegation
    # ==========================================

    def clear_view(self):
        self.render_engine.clear_view()

    def reset_camera(self):
        self.render_engine.reset_camera()

    def render_mesh(self, reset_view=True):
        self.render_engine.render_mesh(reset_view=reset_view)

    def render_volume(self, reset_view=True):
        self.render_engine.render_volume(reset_view=reset_view)

    def render_slices(self, reset_view=True):
        self.render_engine.render_slices(reset_view=reset_view)

    def render_isosurface_auto(self, reset_view=True):
        self.render_engine.render_isosurface_auto(reset_view=reset_view)

    def render_isosurface(self, threshold=300, reset_view=True):
        self.render_engine.render_isosurface(threshold=threshold, reset_view=reset_view)

    # ==========================================
    # Clip Plane Methods
    # ==========================================

    def _on_colormap_changed(self, text):
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=False)
        else:
            self.trigger_render()

    def _on_clim_changed_fast(self):
        """Fast colormap range update without full re-render."""
        if self.active_view_mode == 'volume':
            clim = self.params_panel.get_current_values().get('clim', [0, 1000])
            # Try fast update first, fall back to full render if needed
            if not self.render_engine.update_clim_fast(clim):
                self.render_volume(reset_view=False)
        elif self.active_view_mode == 'slices':
            # Slices also use clim, trigger delayed render
            self.trigger_render()

    def _on_apply_clim_clip(self, clim: list):
        """Apply colormap range as permanent data clip (memory-efficient, chunked)."""
        if not hasattr(self, '_data_manager') or not self._data_manager:
            self.update_status("No data to clip")
            return
        
        data = self._data_manager.active_data
        if data is None or data.raw_data is None:
            self.update_status("No active data to clip")
            return
        
        import gc
        import traceback
        
        min_val, max_val = float(clim[0]), float(clim[1])
        current_step = "initialization"
        
        try:
            current_step = "Step 1: Clearing caches"
            self.update_status(f"Clipping: {current_step}...")
            self.plotter.clear()
            if hasattr(self, 'render_engine'):
                self.render_engine.data = None
                self.render_engine.grid = None
                self.render_engine._cached_vol_grid = None
                self.render_engine._lod_pyramid = None
                self.render_engine._iso_cache = {}
                self.render_engine.volume_actor = None
            gc.collect()
            
            current_step = "Step 2: Applying clip in chunks"
            self.update_status(f"Clipping: {current_step}...")
            raw = data.raw_data
            chunk_size = 32
            n_slices = raw.shape[0]
            
            for i in range(0, n_slices, chunk_size):
                end = min(i + chunk_size, n_slices)
                raw[i:end] = raw[i:end].clip(min_val, max_val)
                gc.collect()
            
            current_step = "Step 3: Updating metadata"
            self.update_status(f"Clipping: {current_step}...")
            data.metadata["ClipRange"] = f"[{min_val:.0f}, {max_val:.0f}]"
            if "(Clipped)" not in data.metadata.get("Type", ""):
                data.metadata["Type"] = data.metadata.get("Type", "CT") + " (Clipped)"
            
            current_step = "Step 4: Updating params panel"
            self.update_status(f"Clipping: {current_step}...")
            self.params_panel.set_data_range(int(min_val), int(max_val))
            
            current_step = "Step 5: Recreating grid"
            self.update_status(f"Clipping: {current_step}...")
            self.render_engine.set_data(data)
            
            current_step = "Step 6: Rendering"
            self.update_status(f"Clipping: {current_step}...")
            if self.active_view_mode == 'volume':
                self.render_volume(reset_view=True)
            elif self.active_view_mode == 'slices':
                self.render_slices(reset_view=True)
            else:
                self.render_volume(reset_view=True)
            
            self.update_status(f"Applied clip: [{min_val:.0f}, {max_val:.0f}]")
            gc.collect()
            
        except MemoryError as e:
            error_msg = f"Memory Error at {current_step}: {e}"
            print(f"[RangeClip] {error_msg}")
            print(traceback.format_exc())
            self.update_status(f"Memory Error: {current_step}")
            gc.collect()
            
        except Exception as e:
            error_msg = f"Error at {current_step}: {type(e).__name__}: {e}"
            print(f"[RangeClip] {error_msg}")
            print(traceback.format_exc())
            self.update_status(f"Error: {current_step} - {type(e).__name__}")

    def _on_invert_volume(self):
        """Invert volume values (for extracting pore surfaces instead of object surfaces)."""
        if not hasattr(self, '_data_manager') or not self._data_manager:
            self.update_status("No data to invert")
            return
        
        data = self._data_manager.active_data
        if data is None or data.raw_data is None:
            self.update_status("No active data to invert")
            return
        
        import gc
        import traceback
        
        try:
            self.update_status("Inverting volume...")
            
            # Step 1: Clear caches
            self.plotter.clear()
            if hasattr(self, 'render_engine'):
                self.render_engine.data = None
                self.render_engine.grid = None
                self.render_engine._cached_vol_grid = None
                self.render_engine._lod_pyramid = None
                self.render_engine._iso_cache = {}
                self.render_engine.volume_actor = None
            gc.collect()
            
            # Step 2: Get min/max for inversion
            raw = data.raw_data
            data_min = float(raw.min())
            data_max = float(raw.max())
            
            # Step 3: Invert in chunks: new_val = max - (val - min) = max + min - val
            chunk_size = 32
            n_slices = raw.shape[0]
            invert_offset = data_max + data_min
            
            for i in range(0, n_slices, chunk_size):
                end = min(i + chunk_size, n_slices)
                raw[i:end] = invert_offset - raw[i:end]
                gc.collect()
            
            # Step 4: Update metadata
            if "(Inverted)" not in data.metadata.get("Type", ""):
                data.metadata["Type"] = data.metadata.get("Type", "CT") + " (Inverted)"
            
            # Step 5: Update params panel range AND invert the current clim values
            # Get current clim values before updating range
            current_clim = self.params_panel.get_current_values().get('clim', [data_min, data_max])
            old_min_clim, old_max_clim = current_clim[0], current_clim[1]
            
            # Invert the clim values: new_clim = max + min - old_clim
            new_min_clim = int(invert_offset - old_max_clim)
            new_max_clim = int(invert_offset - old_min_clim)
            
            # Update range first
            self.params_panel.set_data_range(int(data_min), int(data_max))
            
            # Then set the inverted clim values
            self.params_panel.block_signals(True)
            self.params_panel.slider_clim_min.setValue(new_min_clim)
            self.params_panel.slider_clim_max.setValue(new_max_clim)
            self.params_panel.spinbox_clim_min.setValue(new_min_clim)
            self.params_panel.spinbox_clim_max.setValue(new_max_clim)
            self.params_panel.block_signals(False)
            
            # Step 6: Recreate grid
            self.render_engine.set_data(data)
            
            # Step 7: Render
            if self.active_view_mode == 'volume':
                self.render_volume(reset_view=True)
            elif self.active_view_mode == 'slices':
                self.render_slices(reset_view=True)
            else:
                self.render_volume(reset_view=True)
            
            self.update_status(f"Volume inverted. Clim: [{new_min_clim}, {new_max_clim}]")
            gc.collect()
            
        except Exception as e:
            print(f"[InvertVolume] Error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.update_status(f"Invert error: {type(e).__name__}")

    def _on_clip_toggled(self, enabled: bool):
        if enabled:
            self._apply_clip_planes()
        else:
            mode = self.active_view_mode
            if mode == 'volume':
                self.render_volume(reset_view=True)
            elif mode == 'slices':
                self.render_slices(reset_view=True)
            elif mode == 'iso':
                self.render_isosurface_auto(reset_view=True)
            elif mode == 'mesh':
                self.render_mesh(reset_view=True)

    def _apply_clip_planes(self):
        if not hasattr(self, 'clip_panel'):
            return

        clip_vals = self.clip_panel.get_clip_values()
        if not clip_vals['enabled']:
            return

        EPS = 0.005
        for axis in ['x', 'y', 'z']:
            if not clip_vals[f'invert_{axis}']:
                clip_vals[axis] = max(EPS, clip_vals[axis])
            else:
                clip_vals[axis] = min(1.0 - EPS, clip_vals[axis])

        try:
            data_source = None
            mode = self.active_view_mode
            if mode == 'volume':
                data_source = self.grid
            elif mode == 'mesh':
                data_source = self.mesh
            elif mode == 'iso':
                params = self.params_panel.get_current_values()
                thresh = params['threshold']
                data_source = self.render_engine._iso_cache.get(thresh)
                if not data_source and self.grid:
                    data_source = self.grid.cell_data_to_point_data().contour([thresh])

            if data_source is None:
                return

            bounds = data_source.bounds
            x_min, x_max = bounds[0], bounds[0] + (bounds[1] - bounds[0]) * clip_vals['x']
            y_min, y_max = bounds[2], bounds[2] + (bounds[3] - bounds[2]) * clip_vals['y']
            z_min, z_max = bounds[4], bounds[4] + (bounds[5] - bounds[4]) * clip_vals['z']

            if clip_vals['invert_x']:
                x_min, x_max = x_max, bounds[1]
            if clip_vals['invert_y']:
                y_min, y_max = y_max, bounds[3]
            if clip_vals['invert_z']:
                z_min, z_max = z_max, bounds[5]

            clip_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

            self.plotter.clear()
            self.plotter.add_axes()
            params = self.params_panel.get_current_values()

            if mode == 'volume' and self.grid:
                clipped = self.grid.clip_box(clip_bounds, invert=False)
                if clipped.n_cells > 0:
                    self.plotter.add_mesh(clipped, scalars="values", cmap=params['colormap'],
                                          clim=params['clim'], show_scalar_bar=True, opacity=0.5)

            elif mode == 'mesh' and self.mesh:
                clipped = self.mesh.clip_box(clip_bounds, invert=False)
                if clipped.n_points > 0:
                    scalars = "IsPore" if "IsPore" in clipped.array_names else None
                    cmap = ["gray", "red"] if scalars else params['colormap']
                    self.plotter.add_mesh(clipped, scalars=scalars, cmap=cmap, smooth_shading=True)

            elif mode == 'iso' and data_source:
                clipped = data_source.clip_box(clip_bounds, invert=False)
                style_map = {'Surface': 'surface', 'Wireframe': 'wireframe', 'Wireframe + Surface': 'surface'}
                render_style = style_map.get(params['render_style'], 'surface')
                self.plotter.add_mesh(clipped, color=params['solid_color'], style=render_style, smooth_shading=True)

            self.render_engine._apply_custom_lighting(params)
            self.plotter.render()

        except Exception as e:
            print(f"[Clip] Error: {e}")
            self._on_clip_toggled(False)

    # ==========================================
    # ROI Methods
    # ==========================================

    def _on_roi_toggled(self, enabled: bool):
        if enabled:
            if self.grid is None:
                self.roi_panel.enable_checkbox.setChecked(False)
                return
            bounds = self.grid.bounds
            self.plotter.add_box_widget(
                callback=self._on_roi_bounds_changed,
                bounds=bounds, factor=1.0, rotation_enabled=False, color='cyan', use_planes=False
            )
            self.update_status("ROI mode: Drag the box to select region")
        else:
            self.plotter.clear_box_widgets()
            self.roi_panel.update_bounds(None)
            self.update_status("ROI mode disabled")

    def _on_roi_bounds_changed(self, bounds):
        actual_bounds = bounds.bounds if hasattr(bounds, 'bounds') else bounds
        self.roi_panel.update_bounds(actual_bounds)

    def _on_apply_roi(self):
        roi_bounds = self.roi_panel.get_bounds()
        if roi_bounds is None or self.data is None:
            return

        try:
            extracted = self._extract_roi_subvolume(roi_bounds)
            if extracted is not None:
                if self._data_manager is not None:
                    self._data_manager.set_roi_data(extracted)
                self.set_data(extracted)
                self.update_status(f"ROI applied: {extracted.raw_data.shape}")
                self.roi_panel.enable_checkbox.setChecked(False)
                self.plotter.clear_box_widgets()
        except Exception as e:
            print(f"[ROI] Error: {e}")
            self.update_status("ROI extraction failed")

    def _on_reset_roi(self):
        self.plotter.clear_box_widgets()
        if self._data_manager is not None:
            self._data_manager.clear_roi()
            if self._data_manager.raw_ct_data is not None:
                self.set_data(self._data_manager.raw_ct_data)
        self.update_status("ROI reset")

    def _extract_roi_subvolume(self, bounds) -> Optional[VolumeData]:
        if self.data is None or self.data.raw_data is None:
            return None

        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin

        i_start = max(0, int((bounds[0] - origin[0]) / spacing[0]))
        i_end = min(raw.shape[0], int((bounds[1] - origin[0]) / spacing[0]))
        j_start = max(0, int((bounds[2] - origin[1]) / spacing[1]))
        j_end = min(raw.shape[1], int((bounds[3] - origin[1]) / spacing[1]))
        k_start = max(0, int((bounds[4] - origin[2]) / spacing[2]))
        k_end = min(raw.shape[2], int((bounds[5] - origin[2]) / spacing[2]))

        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end]

        if sub_data.size == 0:
            return None

        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )

        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Extract ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds

        return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=new_metadata)

    # ==========================================
    # Mouse Probe
    # ==========================================

    def _setup_mouse_probe(self):
        self.plotter.iren.add_observer("MouseMoveEvent", self._on_mouse_move)

    def _on_mouse_move(self, obj, event):
        if self.data is None or self.active_view_mode != 'slices':
            return

        try:
            if hasattr(obj, "GetEventPosition"):
                pos = obj.GetEventPosition()
            elif hasattr(obj, "get_event_position"):
                pos = obj.get_event_position()
            else:
                pos = self.plotter.iren.GetEventPosition()

            import vtk
            picker = vtk.vtkPointPicker()
            picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

            if picker.GetViewProp() is None or self.data.raw_data is None:
                return

            world_pos = picker.GetPickPosition()
            ox, oy, oz = self.data.origin
            sx, sy, sz = self.data.spacing

            raw_z = int(round((world_pos[0] - ox) / sx))
            raw_y = int(round((world_pos[1] - oy) / sy))
            raw_x = int(round((world_pos[2] - oz) / sz))

            shape = self.data.raw_data.shape
            if 0 <= raw_z < shape[0] and 0 <= raw_y < shape[1] and 0 <= raw_x < shape[2]:
                val = self.data.raw_data[raw_z, raw_y, raw_x]
                self.status_bar.showMessage(
                    f"📍 Pos: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}) | "
                    f"Indices: [{raw_z}, {raw_y}, {raw_x}] | 💡 HU: {val:.1f}"
                )
        except Exception:
            pass
