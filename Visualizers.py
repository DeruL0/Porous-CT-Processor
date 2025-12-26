import numpy as np
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGroupBox, QFrame, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from typing import Optional
from Core import BaseVisualizer, VolumeData
from GUI import VisualizationModePanel, RenderingParametersPanel, InfoPanel


# Fix for metaclass conflict
class VisualizerMeta(type(QMainWindow), type(BaseVisualizer)):
    pass


class GuiVisualizer(QMainWindow, BaseVisualizer, metaclass=VisualizerMeta):
    """
    PyQt5-based GUI Visualizer with embedded PyVista 3D viewer.
    Designed for Porous Media and Industrial CT Analysis.

    Refactored: UI Logic moved to GUI.py. This class focuses on PyVista rendering.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid = None
        self.current_actors = []  # Track current visualization actors

        # State tracking for active view mode
        self.active_view_mode = None

        # Window setup
        self.setWindowTitle("Porous Media Analysis Suite (Micro-CT)")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel: Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Right panel: 3D Viewer
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Structure Viewer"
        )
        main_layout.addWidget(self.plotter.app_window, stretch=3)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan to begin.")

    def _create_control_panel(self):
        """Create the left control panel using components from GUI.py"""
        panel = QWidget()
        panel.setMaximumWidth(400)

        # Save layout reference so we can add custom panels later via add_custom_panel
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setSpacing(10)

        # Title
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Separator
        layout.addWidget(self._create_separator())

        # 1. Data Info Group (Using reused InfoPanel from GUI would be ideal,
        # but for now we keep local logic if simpler, OR use the InfoPanel class)
        # Let's switch to the GUI.InfoPanel for consistency.
        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel)

        # 2. Visualization Mode Group (From GUI.py)
        self.mode_panel = VisualizationModePanel()
        self.mode_panel.volume_clicked.connect(lambda: self.render_volume(reset_view=True))
        self.mode_panel.slices_clicked.connect(self.render_slices)
        self.mode_panel.iso_clicked.connect(self.render_isosurface_auto)
        self.mode_panel.clear_clicked.connect(self.clear_view)
        self.mode_panel.reset_camera_clicked.connect(self.reset_camera)
        layout.addWidget(self.mode_panel)

        # 3. Rendering Parameters Group (From GUI.py)
        self.params_panel = RenderingParametersPanel()
        # Connect signals
        self.params_panel.threshold_changed.connect(
            lambda v: None)  # Label handled internally, render on demand usually
        # But we might want live update or just store value?
        # For now, the render methods pull values from the panel getters or we use the passed value.
        # Let's just trigger updates where appropriate.

        self.params_panel.threshold_changed.connect(lambda v: None)  # Render button handles this
        self.params_panel.colormap_changed.connect(self._on_parameter_change)
        self.params_panel.solid_color_changed.connect(self._on_parameter_change)
        self.params_panel.opacity_changed.connect(self._on_parameter_change)
        self.params_panel.light_angle_changed.connect(self._on_parameter_change)
        self.params_panel.coloring_mode_changed.connect(self._on_parameter_change)
        self.params_panel.render_style_changed.connect(self._on_parameter_change)

        layout.addWidget(self.params_panel)

        # Stretch to push everything to top
        layout.addStretch()

        return panel

    def add_custom_panel(self, panel: QWidget, index: int = 2):
        """
        Allows the Controller to inject a custom panel (e.g. Workflow/Processing buttons)
        into the control layout without accessing the layout directly.
        """
        if hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def set_data(self, data: VolumeData):
        """Set the data to visualize"""
        self.data = data
        self._create_grid()

        # Update Info Panel
        if self.data:
            data_type = self.data.metadata.get('Type', 'Original Scan')
            self.info_panel.update_info(data_type, self.data.dimensions, self.data.spacing, self.data.metadata)

            # Smart default for solid color (Update GUI)
            target_color = "red" if ("Pore" in data_type or "Network" in data_type) else "ivory"
            idx = self.params_panel.solid_color_combo.findText(target_color)
            if idx >= 0: self.params_panel.solid_color_combo.setCurrentIndex(idx)

        self.update_status("Scan loaded successfully. Choose a visualization mode.")
        self.render_volume(reset_view=True)

    def _create_grid(self):
        """Create PyVista grid from volume data"""
        if not self.data:
            return

        grid = pv.ImageData()
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def clear_view(self):
        """Clear all actors from the viewer"""
        self.plotter.clear()
        self.plotter.add_axes()
        self.active_view_mode = None
        self.params_panel.set_mode(None)
        self.update_status("View cleared.")

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.update_status("Camera reset.")

    def update_status(self, message: str):
        self.status_bar.showMessage(message)

    def show(self):
        super().show()
        self.plotter.app_window.show()

    # ==========================================
    # Visualization Logic
    # ==========================================

    def render_volume(self, reset_view=True):
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status("Rendering volume...")
        self.clear_view()
        self.active_view_mode = 'volume'
        self.params_panel.set_mode('volume')

        self.plotter.renderer.remove_all_lights()
        self.plotter.enable_lightkit()

        vol_grid = self.grid.cell_data_to_point_data()

        # Get params from GUI panel
        params = self.params_panel.get_current_values()

        self.plotter.add_volume(vol_grid, cmap=params['colormap'], opacity=params['opacity'], shade=False)
        self.plotter.add_axes()

        if reset_view:
            self.plotter.reset_camera()
            self.plotter.view_isometric()

        self.update_status(f"Volume rendered.")

    def render_slices(self):
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status("Rendering orthogonal slices...")
        self.clear_view()
        self.active_view_mode = 'slices'
        self.params_panel.set_mode('slices')

        self.plotter.renderer.remove_all_lights()
        self.plotter.enable_lightkit()

        params = self.params_panel.get_current_values()

        slices = self.grid.slice_orthogonal()
        self.plotter.add_mesh(slices, cmap=params['colormap'])
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

        self.update_status(f"Orthogonal slices rendered.")

    def render_isosurface_auto(self):
        # Triggered by button click
        params = self.params_panel.get_current_values()
        self.render_isosurface(threshold=params['threshold'])

    def render_isosurface(self, threshold=300, reset_view=True):
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status(f"Extracting isosurface at threshold {threshold}...")

        min_val, max_val = self.data.raw_data.min(), self.data.raw_data.max()
        if threshold < min_val or threshold > max_val:
            QMessageBox.warning(self, "Invalid Threshold", f"Threshold {threshold} is out of data range.")
            return

        self.clear_view()
        self.active_view_mode = 'iso'
        self.params_panel.set_mode('iso')

        params = self.params_panel.get_current_values()

        # Lighting
        self.plotter.renderer.remove_all_lights()
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.3))

        center = self.grid.center
        radius = self.grid.length * 1.5
        azimuth_deg = params['light_angle']
        elevation_deg = 60

        theta = math.radians(azimuth_deg)
        phi = math.radians(90 - elevation_deg)
        lx = center[0] + radius * math.sin(phi) * math.cos(theta)
        ly = center[1] + radius * math.sin(phi) * math.sin(theta)
        lz = center[2] + radius * math.cos(phi)

        main_light = pv.Light(position=(lx, ly, lz), focal_point=center, intensity=0.8)
        main_light.positional = True
        self.plotter.add_light(main_light)

        # Contouring
        grid_points = self.grid.cell_data_to_point_data()
        try:
            contours = grid_points.contour(isosurfaces=[threshold])

            color_mode = params['coloring_mode']
            cmap = params['colormap']
            solid_color = params['solid_color']

            render_style = 'surface'
            show_edges = False
            if params['render_style'] == 'Wireframe':
                render_style = 'wireframe'
            elif params['render_style'] == 'Wireframe + Surface':
                show_edges = True

            # Rendering
            if color_mode == 'Depth (Z-Axis)':
                contours["Elevation"] = contours.points[:, 2]
                self.plotter.add_mesh(contours, scalars="Elevation", cmap=cmap, style=render_style,
                                      show_edges=show_edges, smooth_shading=True, specular=0.4)
                self.plotter.add_scalar_bar("Elevation (Z)", vertical=True)

            elif color_mode == 'Radial (Center Dist)':
                center_pt = contours.center
                dist = np.linalg.norm(contours.points - center_pt, axis=1)
                contours["RadialDistance"] = dist
                self.plotter.add_mesh(contours, scalars="RadialDistance", cmap=cmap, style=render_style,
                                      show_edges=show_edges, smooth_shading=True, specular=0.4)
                self.plotter.add_scalar_bar("Radial Dist", vertical=True)

            else:  # Solid Color
                self.plotter.add_mesh(contours, color=solid_color, style=render_style, show_edges=show_edges,
                                      smooth_shading=True, specular=0.6, specular_power=15, diffuse=0.8, ambient=0.2)

            self.plotter.add_axes()

            if reset_view:
                self.plotter.reset_camera()
                self.plotter.view_isometric()

            self.update_status(f"Isosurface rendered ({color_mode}).")
        except Exception as e:
            print(f"Isosurface Error: {e}")
            self.update_status("Isosurface generation failed.")

    def _on_parameter_change(self, _):
        """Unified handler for parameter changes from GUI"""
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=False)
        elif self.active_view_mode == 'slices':
            self.render_slices()
        elif self.active_view_mode == 'iso':
            # Need current threshold
            params = self.params_panel.get_current_values()
            self.render_isosurface(threshold=params['threshold'], reset_view=False)

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()