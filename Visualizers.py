import math
from typing import Optional, Dict, Any, List

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QMessageBox, QStatusBar, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

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
        self._iso_cache: Dict[int, pv.PolyData] = {}
        
        # Cache for volume grid to optimize updates
        self._cached_vol_grid: Optional[pv.PolyData] = None
        self._cached_vol_grid_source: Optional[int] = None

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

        # Left Panel (Controls) - with scroll area
        left_scroll = self._create_scrollable_panel()
        main_layout.addWidget(left_scroll, stretch=1)

        # Center Panel (3D Canvas)
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Structure Viewer"
        )
        main_layout.addWidget(self.plotter.app_window, stretch=3)
        
        # Right Panel (Info & Statistics) - with scroll area
        right_scroll = self._create_info_panel()
        main_layout.addWidget(right_scroll, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")

    def _create_scrollable_panel(self) -> QWidget:
        """Create left control panel with scroll support."""
        from PyQt5.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)
        
        panel = QWidget()
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setSpacing(10)

        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())

        self.mode_panel = VisualizationModePanel()
        self.mode_panel.volume_clicked.connect(lambda: self.render_volume(reset_view=True))
        self.mode_panel.slices_clicked.connect(lambda: self.render_slices(reset_view=True))
        self.mode_panel.iso_clicked.connect(self.render_isosurface_auto)
        self.mode_panel.clear_clicked.connect(self.clear_view)
        self.mode_panel.reset_camera_clicked.connect(self.reset_camera)
        layout.addWidget(self.mode_panel)

        self.params_panel = RenderingParametersPanel()

        # Standard rendering triggers (Timer based)
        # Standard rendering triggers (Timer based)
        for signal in [self.params_panel.solid_color_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.render_style_changed,
                       self.params_panel.threshold_changed,
                       self.params_panel.slice_position_changed]:
            signal.connect(self.trigger_render)

        # OPTIMIZATION: Immediate updates for Volume Transfer Function
        self.params_panel.opacity_changed.connect(lambda: self.render_volume(reset_view=False))
        self.params_panel.clim_changed.connect(lambda: self.render_volume(reset_view=False))
        
        # Smart dispatch for colormap (Immediate for Volume, Delayed for others)
        self.params_panel.colormap_changed.connect(self._on_colormap_changed)

        layout.addWidget(self.params_panel)
        layout.addStretch()
        
        scroll_area.setWidget(panel)
        return scroll_area
    
    def _create_info_panel(self) -> QWidget:
        """Create right info/statistics panel with scroll support."""
        from PyQt5.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        title = QLabel("Information")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())
        
        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel)
        
        layout.addStretch()
        
        scroll_area.setWidget(panel)
        return scroll_area

    def add_custom_panel(self, panel: QWidget, index: int = 2, side: str = 'left'):
        """
        Add a custom panel to either left (controls) or right (info) sidebar.
        
        Args:
            panel: Widget to add
            index: Position index in the layout
            side: 'left' for control panel, 'right' for info panel
        """
        if side == 'left' and hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)
        elif side == 'right':
            # Find the right panel's layout
            # The right scroll area's widget's layout
            right_scroll = self.centralWidget().layout().itemAt(2).widget()  # Third item is right panel
            if isinstance(right_scroll, QScrollArea):
                right_widget = right_scroll.widget()
                if right_widget:
                    right_layout = right_widget.layout()
                    # Insert before the final stretch
                    right_layout.insertWidget(right_layout.count() - 1, panel)

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
        self._cached_vol_grid = None # Clear volume cache

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

        # Check if we have pore radius data for size-based coloring
        pore_radii = self.data.metadata.get("PoreRadii") if self.data else None
        
        # Render Pores vs Throats with optional size coloring
        if "IsPore" in self.mesh.array_names:
            # Check if user wants size-based coloring (future: add UI toggle)
            # For now, automatically use size coloring if data available
            if pore_radii and len(pore_radii) > 0:
                # Color by pore size
                # Need to map radii to mesh points
                # Extract IsPore field to identify pore points
                is_pore = self.mesh["IsPore"]
                
                # Create a size scalar field for the mesh
                # This is a simplified approach - assigns radius to pore sphere vertices
                # More sophisticated: assign based on point ID mapping
                size_scalars = np.zeros(self.mesh.n_points)
                
                # Try to get radius from point cloud if available
                if "radius" in self.mesh.point_data:
                    # Use original radius data
                    for i in range(self.mesh.n_points):
                        if is_pore[i] == 1:
                            size_scalars[i] = self.mesh.point_data["radius"][i]
                else:
                    # Fallback: assign based on IsPore
                    pore_indices = np.where(is_pore == 1)[0]
                    if len(pore_indices) > 0 and len(pore_radii) > 0:
                        # Simple assignment
                        avg_radius = np.mean(pore_radii)
                        size_scalars[pore_indices] = avg_radius
                
                self.mesh["PoreSize"] = size_scalars
                
                self.plotter.add_mesh(
                    self.mesh,
                    scalars="PoreSize",
                    cmap="viridis",
                    show_scalar_bar=True,
                    scalar_bar_args={'title': 'Pore Radius (mm)'},
                    smooth_shading=True,
                    specular=0.5
                )
            else:
                # Default binary coloring
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

            # Cache the point data grid to avoid re-computation on every update
            if not hasattr(self, '_cached_vol_grid') or self._cached_vol_grid is None or self._cached_vol_grid_source != id(self.grid):
                self._cached_vol_grid = self.grid.cell_data_to_point_data()
                self._cached_vol_grid_source = id(self.grid)
            
            vol_grid = self._cached_vol_grid
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
            # OPTIMIZATION: Fast Update without full clearing
            self.update_status("Updating volume properties...")
            params = self.params_panel.get_current_values()
            
        # Use cached grid
            vol_grid = self._cached_vol_grid if hasattr(self, '_cached_vol_grid') else self.grid.cell_data_to_point_data()

            # STRATEGY: Double Buffer to prevent flickering
            # Add the NEW actor before removing the OLD one.
            # IMPORTANT: Set render=False to prevent intermediate frames (flashing)
            old_actor = self.volume_actor
            
            new_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params['colormap'],
                opacity=params['opacity'],
                clim=params['clim'],
                shade=False,
                render=False  # Crucial for double buffering to work invisibly
            )
            
            # FORCE UPDATE: Explicitly set scalar range to ensure it takes effect
            # This handles cases where add_volume might default to data range
            if new_actor.mapper:
                new_actor.mapper.scalar_range = params['clim']
            
            print(f"Updating Volume: clim={params['clim']}") # Debug output
            
            if old_actor:
                self.plotter.remove_actor(old_actor, render=False)
            
            self.volume_actor = new_actor
            self.plotter.render()
    
    def _on_colormap_changed(self, text):
        """Handle colormap changes: Immediate for volume, Debounced for others."""
        if self.active_view_mode == 'volume':
            # Volume rendering optimization allows immediate updates
            self.render_volume(reset_view=False)
        else:
            # Other modes might require geometry processing, so debounce
            self.trigger_render()

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

            # Apply light angle if specified
            if 'light_angle' in params and params['light_angle'] is not None:
                angle = params['light_angle']
                # Light position: convert angle to 3D position
                # Angle in degrees, position light around the scene
                import math
                rad = math.radians(angle)
                # Position light in a circle around Z axis
                light_pos = [10 * math.cos(rad), 10 * math.sin(rad), 10]
                self.plotter.remove_all_lights()
                self.plotter.add_light(pv.Light(position=light_pos, intensity=1.0))

            self.plotter.add_axes()
            if reset_view: self.reset_camera()
        except Exception as e:
            print(e)
            self.update_status("Error generating isosurface.")