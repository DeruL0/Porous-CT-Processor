import numpy as np
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGroupBox, QSlider, QComboBox,
                             QFileDialog, QMessageBox, QStatusBar, QFrame, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Optional
from Core import BaseVisualizer, VolumeData


# Fix for metaclass conflict: Create a metaclass that inherits from both
# QMainWindow's metaclass (PyQt5) and BaseVisualizer's metaclass (ABC)
class VisualizerMeta(type(QMainWindow), type(BaseVisualizer)):
    pass


class GuiVisualizer(QMainWindow, BaseVisualizer, metaclass=VisualizerMeta):
    """
    PyQt5-based GUI Visualizer with embedded PyVista 3D viewer.
    Designed for Porous Media and Industrial CT Analysis.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid = None
        self.current_actors = []  # Track current visualization actors

        # State tracking for active view mode
        # Values: 'volume', 'slices', 'iso', or None
        self.active_view_mode = None

        # Window setup
        self.setWindowTitle("Porous CT Analysis Suite")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
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

        # Initial UI State Update
        self._update_parameter_visibility()

    def _create_control_panel(self):
        """Create the left control panel with all controls"""
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

        # Data Info Group
        layout.addWidget(self._create_data_info_group())

        # Visualization Mode Group
        layout.addWidget(self._create_visualization_group())

        # Rendering Parameters Group
        layout.addWidget(self._create_parameters_group())

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
        """Create a horizontal line separator"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _create_data_info_group(self):
        """Create data information display group"""
        group = QGroupBox("Sample Information")
        layout = QVBoxLayout()

        self.data_type_label = QLabel("Type: No data loaded")
        self.data_dim_label = QLabel("Grid: N/A")
        self.data_spacing_label = QLabel("Voxel Size: N/A")
        self.data_meta_label = QLabel("Sample Data: N/A")

        for label in [self.data_type_label, self.data_dim_label,
                      self.data_spacing_label, self.data_meta_label]:
            label.setWordWrap(True)
            layout.addWidget(label)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self):
        """Create visualization mode selection group"""
        group = QGroupBox("Analysis Modes")
        layout = QVBoxLayout()

        # Volume Rendering Button
        self.btn_volume = QPushButton("📊 Volume Rendering")
        self.btn_volume.setMinimumHeight(40)
        self.btn_volume.clicked.connect(lambda: self.render_volume(reset_view=True))
        layout.addWidget(self.btn_volume)

        # Slice View Button
        self.btn_slices = QPushButton("🔳 Orthogonal Slices")
        self.btn_slices.setMinimumHeight(40)
        self.btn_slices.clicked.connect(self.render_slices)
        layout.addWidget(self.btn_slices)

        # Isosurface Button
        self.btn_isosurface = QPushButton("🏔️ Isosurface (Solid/Pore)")
        self.btn_isosurface.setMinimumHeight(40)
        self.btn_isosurface.clicked.connect(self.render_isosurface_auto)
        layout.addWidget(self.btn_isosurface)

        # Separator
        layout.addWidget(self._create_separator())

        # Clear View Button
        self.btn_clear = QPushButton("🗑️ Clear View")
        self.btn_clear.setMinimumHeight(35)
        self.btn_clear.clicked.connect(self.clear_view)
        layout.addWidget(self.btn_clear)

        # Reset Camera Button
        self.btn_reset_camera = QPushButton("🎥 Reset Camera")
        self.btn_reset_camera.setMinimumHeight(35)
        self.btn_reset_camera.clicked.connect(self.reset_camera)
        layout.addWidget(self.btn_reset_camera)

        group.setLayout(layout)
        return group

    def _create_parameters_group(self):
        """Create rendering parameters control group"""
        group = QGroupBox("Rendering Parameters")
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
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
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
        self.coloring_mode_combo.setToolTip("Select how the isosurface is colored to enhance visibility.")
        self.coloring_mode_combo.setCurrentIndex(0)
        self.coloring_mode_combo.currentTextChanged.connect(self._on_coloring_mode_changed)
        layout.addWidget(self.coloring_mode_combo)

        # 3. Colormap selector (Volume, Slices, or Iso-Scalar)
        self.lbl_colormap = QLabel("Colormap:")
        layout.addWidget(self.lbl_colormap)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['bone', 'viridis', 'plasma', 'gray', 'coolwarm', 'jet', 'magma'])
        # Connect change event to our smart handler
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        # 4. Solid Color Selector (Iso-Solid only)
        self.lbl_solid_color = QLabel("Solid Color:")
        layout.addWidget(self.lbl_solid_color)

        self.solid_color_combo = QComboBox()
        self.solid_color_combo.addItems(['ivory', 'red', 'gold', 'lightgray', 'mediumseagreen', 'dodgerblue', 'wheat'])
        # Connect change event to our smart handler
        self.solid_color_combo.currentTextChanged.connect(self._on_solid_color_changed)
        layout.addWidget(self.solid_color_combo)

        # 5. Light Azimuth (Position) Slider (Iso only)
        self.lbl_light_angle = QLabel("Light Source Angle (0-360°):")
        layout.addWidget(self.lbl_light_angle)

        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(0, 360)
        self.light_azimuth_slider.setValue(45)
        self.light_azimuth_slider.setTickInterval(45)
        # Only trigger update if relevant
        self.light_azimuth_slider.valueChanged.connect(self._on_light_angle_changed)
        layout.addWidget(self.light_azimuth_slider)

        # 6. Opacity preset selector (Volume only)
        self.lbl_opacity = QLabel("Opacity Preset:")
        layout.addWidget(self.lbl_opacity)

        self.opacity_combo = QComboBox()
        self.opacity_combo.addItems(['sigmoid', 'sigmoid_10', 'linear', 'linear_r', 'geom', 'geom_r'])
        # Connect change event to our smart handler
        self.opacity_combo.currentTextChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_combo)

        group.setLayout(layout)
        return group

    def _update_parameter_visibility(self):
        """
        Dynamically show/hide controls based on the active view mode.
        """
        mode = self.active_view_mode

        # Helper to toggle visibility of a group of widgets
        def set_visible(widgets, visible):
            for w in widgets:
                w.setVisible(visible)

        # Defaults: Hide everything first, then enable based on logic
        # Or easier: define logic variables

        show_threshold = (mode == 'iso')
        show_coloring_mode = (mode == 'iso')
        show_light_angle = (mode == 'iso')
        show_opacity = (mode == 'volume')

        show_colormap = False
        show_solid_color = False

        if mode == 'volume':
            show_colormap = True
        elif mode == 'slices':
            show_colormap = True
        elif mode == 'iso':
            # Check if using Solid or Scalar coloring
            if self.coloring_mode_combo.currentText() == 'Solid Color':
                show_solid_color = True
            else:
                show_colormap = True

        # Apply visibility
        set_visible([self.lbl_threshold, self.threshold_slider, self.threshold_value_label], show_threshold)
        set_visible([self.lbl_coloring_mode, self.coloring_mode_combo], show_coloring_mode)
        set_visible([self.lbl_light_angle, self.light_azimuth_slider], show_light_angle)
        set_visible([self.lbl_opacity, self.opacity_combo], show_opacity)
        set_visible([self.lbl_colormap, self.colormap_combo], show_colormap)
        set_visible([self.lbl_solid_color, self.solid_color_combo], show_solid_color)

    def set_data(self, data: VolumeData):
        """Set the data to visualize"""
        self.data = data
        self._create_grid()
        self._update_data_info()

        # Smart default for solid color
        data_type = self.data.metadata.get("Type", "")
        target_color = "red" if ("Pore" in data_type or "Network" in data_type) else "ivory"
        idx = self.solid_color_combo.findText(target_color)
        if idx >= 0: self.solid_color_combo.setCurrentIndex(idx)

        self.update_status("Scan loaded successfully. Choose a visualization mode.")

        # Automatically show volume rendering
        self.render_volume(reset_view=True)

    def _create_grid(self):
        """Create PyVista grid from volume data"""
        if not self.data:
            return

        grid = pv.ImageData()
        # Note: PyVista dimensions (x, y, z) vs Numpy shape (z, y, x) logic
        # is handled by how we flatten the array.
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def _update_data_info(self):
        """Update data information labels"""
        if not self.data:
            return

        data_type = self.data.metadata.get('Type', 'Original Scan')
        self.data_type_label.setText(f"Type: {data_type}")
        self.data_dim_label.setText(f"Grid: {self.data.dimensions}")
        self.data_spacing_label.setText(
            f"Voxel Size: ({self.data.spacing[0]:.2f}, "
            f"{self.data.spacing[1]:.2f}, {self.data.spacing[2]:.2f}) mm"
        )

        # Format metadata
        meta_str = "\n".join([f"{k}: {v}" for k, v in self.data.metadata.items()
                              if k != 'Type'][:3])
        self.data_meta_label.setText(f"Sample Data:\n{meta_str}" if meta_str else "Sample Data: None")

    def clear_view(self):
        """Clear all actors from the viewer"""
        self.plotter.clear()
        self.plotter.add_axes()
        self.active_view_mode = None  # Reset active mode
        self._update_parameter_visibility()
        self.update_status("View cleared.")

    def reset_camera(self):
        """Reset camera to default view"""
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.update_status("Camera reset.")

    def update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.showMessage(message)

    def show(self):
        """Show the GUI window"""
        super().show()
        self.plotter.app_window.show()

    # Visualization Methods

    def render_volume(self, reset_view=True):
        """
        Render volume visualization
        :param reset_view: If True, resets camera. If False, keeps current camera position (for updates).
        """
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status("Rendering volume...")
        self.clear_view()
        self.active_view_mode = 'volume'  # Set mode
        self._update_parameter_visibility()

        # Reset lighting to default for volume rendering (best results)
        self.plotter.renderer.remove_all_lights()
        self.plotter.enable_lightkit()

        # Convert cell data to point data
        vol_grid = self.grid.cell_data_to_point_data()

        # Get current colormap and opacity
        cmap = self.colormap_combo.currentText()
        opacity = self.opacity_combo.currentText()

        # Add volume rendering
        self.plotter.add_volume(vol_grid, cmap=cmap, opacity=opacity, shade=False)
        self.plotter.add_axes()

        if reset_view:
            self.plotter.reset_camera()
            self.plotter.view_isometric()

        self.update_status(f"Volume rendered with colormap: {cmap}, opacity: {opacity}")

    def render_slices(self):
        """Render orthogonal slices"""
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status("Rendering orthogonal slices...")
        self.clear_view()
        self.active_view_mode = 'slices'  # Set mode
        self._update_parameter_visibility()

        # Reset lighting
        self.plotter.renderer.remove_all_lights()
        self.plotter.enable_lightkit()

        # Use selected colormap for slices too
        cmap = self.colormap_combo.currentText()

        slices = self.grid.slice_orthogonal()
        self.plotter.add_mesh(slices, cmap=cmap)
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

        self.update_status(f"Orthogonal slices rendered ({cmap}).")

    def render_isosurface_auto(self):
        """Render isosurface with automatic threshold selection"""
        threshold = self.threshold_slider.value()
        self.render_isosurface(threshold=threshold)

    def render_isosurface(self, threshold=300, color=None, opacity=1.0, reset_view=True):
        """
        Render isosurface at given threshold.
        Supports advanced coloring modes (Depth, Radial) and Custom Lighting Position.
        """
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load a sample first.")
            return

        self.update_status(f"Extracting isosurface at threshold {threshold}...")

        # Check if threshold is in valid range
        min_val, max_val = self.data.raw_data.min(), self.data.raw_data.max()
        if threshold < min_val or threshold > max_val:
            QMessageBox.warning(
                self,
                "Invalid Threshold",
                f"Threshold {threshold} is out of data range [{min_val:.0f}, {max_val:.0f}]"
            )
            return

        self.clear_view()
        self.active_view_mode = 'iso'  # Set mode
        self._update_parameter_visibility()

        # --- Custom Lighting Setup for Isosurface ---
        # We replace default lighting with a specific directional light that the user can rotate.
        # This enhances the "3D feel" and reflection movement.
        self.plotter.renderer.remove_all_lights()

        # 1. Soft Fill Light (Headlight - moves with camera)
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=0.3))

        # 2. Main Directional Light (Position controlled by slider)
        # Calculate position relative to data center
        center = self.grid.center
        length = self.grid.length  # Diagonal length
        radius = length * 1.5  # Place light outside the object

        azimuth_deg = self.light_azimuth_slider.value()
        elevation_deg = 60  # Fixed elevation from top

        theta = math.radians(azimuth_deg)
        phi = math.radians(90 - elevation_deg)

        # Spherical to Cartesian conversion
        lx = center[0] + radius * math.sin(phi) * math.cos(theta)
        ly = center[1] + radius * math.sin(phi) * math.sin(theta)
        lz = center[2] + radius * math.cos(phi)

        # Create and add the main light
        main_light = pv.Light(position=(lx, ly, lz), focal_point=center, intensity=0.8)
        main_light.positional = True
        self.plotter.add_light(main_light)

        # --- Isosurface Generation ---

        # Convert cell data to point data for contouring
        grid_points = self.grid.cell_data_to_point_data()

        try:
            contours = grid_points.contour(isosurfaces=[threshold])

            # REMOVED: contours.compute_normals(inplace=True, split_vertices=True)
            # User requested to disable normal computation for cleaner flat look or performance.

            # Get settings
            color_mode = self.coloring_mode_combo.currentText()
            cmap = self.colormap_combo.currentText()

            # --- Logic for Coloring Modes ---

            if color_mode == 'Depth (Z-Axis)':
                # Color by Z-coordinate
                contours["Elevation"] = contours.points[:, 2]

                self.plotter.add_mesh(
                    contours,
                    scalars="Elevation",
                    cmap=cmap,
                    opacity=opacity,
                    smooth_shading=True,
                    specular=0.4
                )
                self.plotter.add_scalar_bar("Elevation (Z)", vertical=True)

            elif color_mode == 'Radial (Center Dist)':
                # Color by distance from the mesh center
                # This highlights the core vs outer shell structure
                center = contours.center
                pts = contours.points
                # Euclidean distance from center
                dist = np.linalg.norm(pts - center, axis=1)
                contours["RadialDistance"] = dist

                self.plotter.add_mesh(
                    contours,
                    scalars="RadialDistance",
                    cmap=cmap,
                    opacity=opacity,
                    smooth_shading=True,
                    specular=0.4
                )
                self.plotter.add_scalar_bar("Radial Dist", vertical=True)

            else:
                # --- Solid Color Mode ---
                # Enhanced lighting to solve "flat look" issues
                if color is None:
                    # Use user selected color from GUI
                    color = self.solid_color_combo.currentText()

                # DISABLED PBR (Physically Based Rendering) as requested
                # Reverted to standard Phong shading
                self.plotter.add_mesh(
                    contours,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    # pbr=False,
                    specular=0.6,  # Standard specular
                    specular_power=15,  # Moderate shininess
                    diffuse=0.8,
                    ambient=0.2,
                    show_edges=False
                )

            self.plotter.add_axes()

            if reset_view:
                self.plotter.reset_camera()
                self.plotter.view_isometric()

            self.update_status(
                f"Isosurface rendered ({color_mode}) at threshold {threshold}. Light Angle: {azimuth_deg}°")
        except Exception as e:
            # QMessageBox.critical(self, "Rendering Error", f"Failed to render isosurface: {str(e)}")
            # Fallback for empty isosurfaces (e.g. threshold too high/low)
            print(f"Isosurface Error: {e}")
            self.update_status("Isosurface generation failed (likely empty result).")

    # Callback methods for parameter changes

    def _on_threshold_changed(self, value):
        """Update threshold value label"""
        self.threshold_value_label.setText(f"Value: {value} Intensity")
        # Optional: We could trigger re-render here if the user wants live update
        # but thresholding is expensive, so keeping it manual (or via button) is safer.

    def _on_colormap_changed(self, colormap):
        """
        Smart Handler: Only updates the active view mode that uses a Colormap.
        """
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=False)
        elif self.active_view_mode == 'slices':
            self.render_slices()
        elif self.active_view_mode == 'iso':
            # Only re-render iso if we are NOT in 'Solid Color' mode
            # because solid color mode doesn't use the colormap.
            if self.coloring_mode_combo.currentText() != 'Solid Color':
                self.render_isosurface(threshold=self.threshold_slider.value(), reset_view=False)

    def _on_solid_color_changed(self, color):
        """
        Smart Handler: Only updates if we are actively viewing an Isosurface in Solid Mode.
        """
        if self.active_view_mode == 'iso':
            if self.coloring_mode_combo.currentText() == 'Solid Color':
                self.render_isosurface(threshold=self.threshold_slider.value(), reset_view=False)

    def _on_opacity_changed(self, opacity):
        """Handle opacity preset change - Only for Volume"""
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=False)

    def _on_light_angle_changed(self, value):
        """Handle light angle change - Only for Isosurface"""
        if self.active_view_mode == 'iso':
            self.render_isosurface(threshold=self.threshold_slider.value(), reset_view=False)

    def _on_coloring_mode_changed(self, mode):
        """Handle coloring mode change - Only for Isosurface"""
        if self.active_view_mode == 'iso':
            self._update_parameter_visibility()  # Need to update visibility because mode changed (e.g. Solid -> Depth)
            self.render_isosurface(threshold=self.threshold_slider.value(), reset_view=False)

    def closeEvent(self, event):
        """Handle window close event"""
        self.plotter.close()
        event.accept()