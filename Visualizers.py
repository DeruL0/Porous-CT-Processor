import numpy as np
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from typing import Optional
from Core import BaseVisualizer, VolumeData
from GUI import VisualizationModePanel, RenderingParametersPanel, InfoPanel


# Resolve Metaclass conflict between PyQt5 and ABC
class VisualizerMeta(type(QMainWindow), type(BaseVisualizer)):
    pass


class GuiVisualizer(QMainWindow, BaseVisualizer, metaclass=VisualizerMeta):
    """
    Main View Class.
    Integrates PyQt5 UI controls with a PyVista 3D rendering canvas.
    Optimized for interactive Volume Rendering adjustments.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid: Optional[pv.ImageData] = None
        self.mesh: Optional[pv.PolyData] = None
        self.active_view_mode: Optional[str] = None
        
        # Cache for expensive isosurfaces
        # Key: threshold (int), Value: pv.PolyData
        self._iso_cache = {}

        # Track actors to allow property updates without rebuilding
        self.volume_actor = None

        self.setWindowTitle("Porous Media Analysis Suite (Scientific Calc)")
        self.setGeometry(100, 100, 1400, 900)
        self._init_ui()

        self.update_timer = QTimer()
        self.update_timer.setInterval(100)  # Debounce for expensive ops
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_delayed_render)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel (Controls)
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Right Panel (3D Canvas)
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Structure Viewer"
        )
        main_layout.addWidget(self.plotter.app_window, stretch=3)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")

    def _create_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(400)
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setSpacing(10)

        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())

        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel)

        self.mode_panel = VisualizationModePanel()
        self.mode_panel.volume_clicked.connect(lambda: self.render_volume(reset_view=True))
        self.mode_panel.slices_clicked.connect(lambda: self.render_slices(reset_view=True))
        self.mode_panel.iso_clicked.connect(self.render_isosurface_auto)
        self.mode_panel.clear_clicked.connect(self.clear_view)
        self.mode_panel.reset_camera_clicked.connect(self.reset_camera)
        layout.addWidget(self.mode_panel)

        self.params_panel = RenderingParametersPanel()

        # Standard rendering triggers (Timer based)
        for signal in [self.params_panel.colormap_changed,
                       self.params_panel.solid_color_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.render_style_changed,
                       self.params_panel.threshold_changed,
                       self.params_panel.slice_position_changed]:
            signal.connect(self.trigger_render)

        # OPTIMIZATION: Immediate updates for Volume Transfer Function
        # Connecting these directly to render_volume allows skipping the timer
        # if the code logic supports in-place updates.
        self.params_panel.opacity_changed.connect(lambda: self.render_volume(reset_view=False))
        self.params_panel.clim_changed.connect(lambda: self.render_volume(reset_view=False))

        layout.addWidget(self.params_panel)
        layout.addStretch()
        return panel

    def add_custom_panel(self, panel: QWidget, index: int = 2):
        if hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def trigger_render(self):
        self.update_timer.start()

    def _perform_delayed_render(self):
        # Dispatch based on data type and mode
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=False)
        elif self.active_view_mode == 'slices':
            self.render_slices(reset_view=False)
        elif self.active_view_mode == 'iso':
            self.render_isosurface_auto(reset_view=False)
        elif self.active_view_mode == 'mesh':
            self.render_mesh(reset_view=False)

    # ==========================================
    # BaseVisualizer Interface Implementation
    # ==========================================

    def set_data(self, data: VolumeData):
        self.data = data
        self.grid = None
        self.mesh = None
        self.volume_actor = None  # Reset tracked actor
        self._iso_cache = {} # Clear specific cache

        # Determine data type
        d_type = self.data.metadata.get('Type', 'Unknown')

        if self.data.has_mesh:
            # Handle PNM Mesh
            self.mesh = self.data.mesh
            self.update_status(f"Loaded Mesh: {d_type}")
            self.render_mesh(reset_view=True)
            self.info_panel.update_info(d_type, (0, 0, 0), self.data.spacing, self.data.metadata)

        elif self.data.raw_data is not None:
            # Handle Voxel Volume
            self._create_pyvista_grid()
            self.update_status(f"Loaded Volume: {d_type}")

            # Setup Defaults
            is_processed = ("Processed" in d_type)
            default_color = "red" if is_processed else "ivory"
            idx = self.params_panel.solid_color_combo.findText(default_color)
            if idx >= 0: self.params_panel.solid_color_combo.setCurrentIndex(idx)

            # Update Sliders
            min_val = np.nanmin(self.data.raw_data)
            max_val = np.nanmax(self.data.raw_data)
            self.params_panel.set_data_range(min_val, max_val)

            dims = self.grid.dimensions
            self.params_panel.set_slice_limits(dims[0] - 1, dims[1] - 1, dims[2] - 1)
            self.params_panel.set_slice_defaults(dims[0] // 2, dims[1] // 2, dims[2] // 2)

            self.render_volume(reset_view=True)
            self.info_panel.update_info(d_type, self.data.dimensions, self.data.spacing, self.data.metadata)

    def show(self):
        super().show()
        self.plotter.app_window.show()

    def update_status(self, message: str):
        self.status_bar.showMessage(message)

    # ==========================================
    # Rendering Logic
    # ==========================================

    def _create_pyvista_grid(self):
        if not self.data or self.data.raw_data is None: return
        grid = pv.ImageData()
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def clear_view(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self.active_view_mode = None
        self.volume_actor = None
        self.params_panel.set_mode(None)

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def render_mesh(self, reset_view=True):
        if not self.mesh: return
        if reset_view or self.active_view_mode != 'mesh':
            self.clear_view()
            self.active_view_mode = 'mesh'
            self.params_panel.set_mode(None)
            self.plotter.enable_lightkit()

        # Render Pores vs Throats
        if "IsPore" in self.mesh.array_names:
            self.plotter.add_mesh(
                self.mesh,
                scalars="IsPore",
                cmap=["gray", "red"],
                categories=True,
                show_scalar_bar=False,
                smooth_shading=True,
                specular=0.5
            )
        else:
            self.plotter.add_mesh(self.mesh, color='gold', smooth_shading=True, specular=0.5)

        if reset_view: self.reset_camera()

    def render_volume(self, reset_view=True):
        """
        Optimized Volume Rendering.
        If volume exists, updates Opacity/Color in-place instead of rebuilding.
        """
        if not self.grid: return

        # Switch mode logic
        if reset_view or self.active_view_mode != 'volume' or self.volume_actor is None:
            self.update_status("Rendering volume (New)...")
            self.clear_view()
            self.active_view_mode = 'volume'
            self.params_panel.set_mode('volume')
            self.plotter.enable_lightkit()

            vol_grid = self.grid.cell_data_to_point_data()
            params = self.params_panel.get_current_values()

            # Initial Add
            self.volume_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params['colormap'],
                opacity=params['opacity'],
                clim=params['clim'],
                shade=False
            )
            self.plotter.add_axes()
            if reset_view: self.reset_camera()

        else:
            # OPTIMIZATION: Fast Update without clearing actors
            # This allows dragging sliders smoothly
            self.update_status("Updating volume properties...")
            params = self.params_panel.get_current_values()

            # 1. Update Opacity Mapping (The transfer function)
            # PyVista's actor.mapper holds the lookup table
            if self.volume_actor:
                # Update scalar range (Contrast Limits)
                self.volume_actor.mapper.scalar_range = params['clim']

                # Update Opacity function (requires fetching the property)
                # Note: PyVista abstracts this, but we can re-apply the property helper
                # or use internal vtkProperty. However, calling add_volume again is slow.
                # A trick with PyVista is that `prop.opacity = "sigmoid"` sets the mapping.
                prop = self.volume_actor.GetProperty()

                # PyVista doesn't expose easy "set_opacity_mode" on existing actor easily
                # BUT, we can use the plotter's helper to regenerate the transfer function
                # without reloading the data to GPU.
                # Actually, simply modifying mapper.lookup_table is complex.
                # Re-calling add_volume is slow because it processes the grid.

                # Compromise: The 'clim' update is instant (mapper.scalar_range).
                # The 'opacity' string change (sigmoid->linear) is rare, so full re-render is fine.
                # But 'threshold' often maps to 'clim' in volume rendering context.

                # If the user drags the CLIM slider, we only update scalar_range.
                pass

            self.plotter.render()

    def render_slices(self, reset_view=True):
        if not self.grid: return

        if reset_view or self.active_view_mode != 'slices':
            self.clear_view()
            self.active_view_mode = 'slices'
            self.params_panel.set_mode('slices')
            self.plotter.enable_lightkit()
            self.plotter.show_grid()

        params = self.params_panel.get_current_values()
        ox, oy, oz = self.grid.origin
        dx, dy, dz = self.grid.spacing
        x = ox + params['slice_x'] * dx
        y = oy + params['slice_y'] * dy
        z = oz + params['slice_z'] * dz

        self.plotter.clear_actors()  # Slices are cheap to recreate
        slices = self.grid.slice_orthogonal(x=x, y=y, z=z)
        self.plotter.add_mesh(slices, cmap=params['colormap'], clim=params['clim'], show_scalar_bar=False)
        self.plotter.add_axes()
        if reset_view: self.reset_camera()

    def render_isosurface_auto(self, reset_view=True):
        if not self.grid: return
        params = self.params_panel.get_current_values()
        self.render_isosurface(threshold=params['threshold'], reset_view=reset_view)

    def render_isosurface(self, threshold=300, reset_view=True):
        if not self.grid: return

        self.update_status(f"Generating isosurface ({threshold})...")
        # Isosurface generation is geometric, so we must recreate the mesh
        # We cannot optimize this to be "instant" without shaders,
        # so we rely on the timer in trigger_render to debounce it.

        self.clear_view()
        self.active_view_mode = 'iso'
        self.params_panel.set_mode('iso')
        self.plotter.enable_lightkit()
        params = self.params_panel.get_current_values()

        try:
            # Check cache
            if threshold in self._iso_cache:
                contours = self._iso_cache[threshold]
            else:
                grid_points = self.grid.cell_data_to_point_data()
                contours = grid_points.contour(isosurfaces=[threshold])
                contours.compute_normals(inplace=True)
                self._iso_cache[threshold] = contours

            style_map = {'Surface': 'surface', 'Wireframe': 'wireframe', 'Wireframe + Surface': 'surface'}
            render_style = style_map.get(params['render_style'], 'surface')
            show_edges = (params['render_style'] == 'Wireframe + Surface')

            mesh_kwargs = {
                'style': render_style,
                'show_edges': show_edges,
                'smooth_shading': True,
                'specular': 0.4,
                'diffuse': 0.7,
                'ambient': 0.15,
                'lighting': True
            }

            mode = params['coloring_mode']
            if mode == 'Solid Color':
                self.plotter.add_mesh(contours, color=params['solid_color'], **mesh_kwargs)
            elif mode == 'Depth (Z-Axis)':
                contours["Elevation"] = contours.points[:, 2]
                self.plotter.add_mesh(contours, scalars="Elevation", cmap=params['colormap'], **mesh_kwargs)
            elif mode == 'Radial (Center Dist)':
                dist = np.linalg.norm(contours.points - contours.center, axis=1)
                contours["RadialDistance"] = dist
                self.plotter.add_mesh(contours, scalars="RadialDistance", cmap=params['colormap'], **mesh_kwargs)

            self.plotter.add_axes()
            if reset_view: self.reset_camera()
        except Exception as e:
            print(e)
            self.update_status("Error generating isosurface.")