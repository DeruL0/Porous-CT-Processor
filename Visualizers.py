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
    Now supports both Voxel Grid and Mesh Visualization.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid: Optional[pv.ImageData] = None
        self.mesh: Optional[pv.PolyData] = None  # Support for PNM Mesh
        self.active_view_mode: Optional[str] = None

        self.setWindowTitle("Porous Media Analysis Suite (Scientific Calc)")
        self.setGeometry(100, 100, 1400, 900)
        self._init_ui()

        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
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
        # Connect parameters signals...
        for signal in [self.params_panel.colormap_changed,
                       self.params_panel.solid_color_changed,
                       self.params_panel.opacity_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.render_style_changed,
                       self.params_panel.clim_changed,
                       self.params_panel.threshold_changed]:
            signal.connect(self.trigger_render)

        self.params_panel.slice_position_changed.connect(self.trigger_render)
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

    def trigger_render(self, *args):
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
        self.params_panel.set_mode(None)

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def render_mesh(self, reset_view=True):
        """Renders the PNM Mesh (PolyData)."""
        if not self.mesh: return

        if reset_view or self.active_view_mode != 'mesh':
            self.clear_view()
            self.active_view_mode = 'mesh'
            self.params_panel.set_mode(None)  # Disable standard volume controls
            self.plotter.enable_lightkit()

        # Render Pores vs Throats
        # We use the 'IsPore' scalar we created in the Processor
        if "IsPore" in self.mesh.array_names:
            self.plotter.add_mesh(
                self.mesh,
                scalars="IsPore",
                cmap=["gray", "red"],  # 0=Throat(Gray), 1=Pore(Red)
                categories=True,
                show_scalar_bar=False,
                smooth_shading=True,
                specular=0.5,
                diffuse=0.8,
                ambient=0.15
            )
        else:
            self.plotter.add_mesh(
                self.mesh,
                color='gold',
                smooth_shading=True,
                specular=0.5
            )

        if reset_view: self.reset_camera()

    def render_volume(self, reset_view=True):
        if not self.grid: return  # Cannot render volume if no grid

        self.update_status("Rendering volume...")
        if reset_view or self.active_view_mode != 'volume':
            self.clear_view()
            self.active_view_mode = 'volume'
            self.params_panel.set_mode('volume')
            self.plotter.enable_lightkit()

        vol_grid = self.grid.cell_data_to_point_data()
        params = self.params_panel.get_current_values()
        self.plotter.clear_actors()
        self.plotter.add_volume(
            vol_grid,
            cmap=params['colormap'],
            opacity=params['opacity'],
            clim=params['clim'],
            shade=False
        )
        self.plotter.add_axes()
        if reset_view: self.reset_camera()

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

        slices = self.grid.slice_orthogonal(x=x, y=y, z=z)
        self.plotter.add_mesh(slices, cmap=params['colormap'], clim=params['clim'], show_scalar_bar=False)
        if reset_view: self.reset_camera()

    def render_isosurface_auto(self, reset_view=True):
        if not self.grid: return
        params = self.params_panel.get_current_values()
        self.render_isosurface(threshold=params['threshold'], reset_view=reset_view)

    def render_isosurface(self, threshold=300, reset_view=True):
        if not self.grid: return

        self.update_status(f"Generating isosurface (Threshold: {threshold})...")
        self.clear_view()
        self.active_view_mode = 'iso'
        self.params_panel.set_mode('iso')

        # Ensure lighting is enabled for shading to work
        self.plotter.enable_lightkit()

        params = self.params_panel.get_current_values()

        try:
            grid_points = self.grid.cell_data_to_point_data()
            contours = grid_points.contour(isosurfaces=[threshold])

            # Critical: Compute normals to allow for smooth shading
            contours.compute_normals(inplace=True)

            # Determine Rendering Style
            style_map = {'Surface': 'surface', 'Wireframe': 'wireframe', 'Wireframe + Surface': 'surface'}
            render_style = style_map.get(params['render_style'], 'surface')
            show_edges = (params['render_style'] == 'Wireframe + Surface')

            # Improved rendering properties combined with style logic
            mesh_kwargs = {
                'style': render_style,
                'show_edges': show_edges,
                'smooth_shading': True,  # Gouraud shading
                'specular': 0.4,  # Intensity of specular highlights
                'specular_power': 20,  # Shininess
                'diffuse': 0.7,  # Diffuse reflection
                'ambient': 0.15,  # Ambient light
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