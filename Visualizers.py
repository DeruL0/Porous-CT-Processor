import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGroupBox, QSlider, QComboBox,
                             QFileDialog, QMessageBox, QStatusBar, QFrame)
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
    Replaces command-line interface with an interactive graphical interface.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid = None
        self.current_actors = []  # Track current visualization actors

        # Window setup
        self.setWindowTitle("Medical Imaging Visualization Suite")
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
        # FIXED: window_size must be a tuple, not a list
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Visualization"
        )
        main_layout.addWidget(self.plotter.app_window, stretch=3)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load data to begin.")

    def _create_control_panel(self):
        """Create the left control panel with all controls"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
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

    def _create_separator(self):
        """Create a horizontal line separator"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _create_data_info_group(self):
        """Create data information display group"""
        group = QGroupBox("Data Information")
        layout = QVBoxLayout()

        self.data_type_label = QLabel("Type: No data loaded")
        self.data_dim_label = QLabel("Dimensions: N/A")
        self.data_spacing_label = QLabel("Spacing: N/A")
        self.data_meta_label = QLabel("Metadata: N/A")

        for label in [self.data_type_label, self.data_dim_label,
                      self.data_spacing_label, self.data_meta_label]:
            label.setWordWrap(True)
            layout.addWidget(label)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self):
        """Create visualization mode selection group"""
        group = QGroupBox("Visualization Modes")
        layout = QVBoxLayout()

        # Volume Rendering Button
        self.btn_volume = QPushButton("🧊 Volume Rendering")
        self.btn_volume.setMinimumHeight(40)
        self.btn_volume.clicked.connect(self.render_volume)
        layout.addWidget(self.btn_volume)

        # Slice View Button
        self.btn_slices = QPushButton("🔪 Orthogonal Slices")
        self.btn_slices.setMinimumHeight(40)
        self.btn_slices.clicked.connect(self.render_slices)
        layout.addWidget(self.btn_slices)

        # Isosurface Button
        self.btn_isosurface = QPushButton("🌐 Isosurface")
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

        # Threshold slider for isosurface
        threshold_label = QLabel("Isosurface Threshold:")
        layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(-1000)
        self.threshold_slider.setMaximum(2000)
        self.threshold_slider.setValue(300)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(200)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("Value: 300 HU")
        self.threshold_value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.threshold_value_label)

        # Colormap selector
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['bone', 'viridis', 'plasma', 'gray', 'coolwarm', 'jet'])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        # Opacity preset selector
        layout.addWidget(QLabel("Opacity Preset:"))
        self.opacity_combo = QComboBox()
        self.opacity_combo.addItems(['sigmoid', 'sigmoid_10', 'linear', 'linear_r', 'geom', 'geom_r'])
        self.opacity_combo.currentTextChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_combo)

        group.setLayout(layout)
        return group

    def set_data(self, data: VolumeData):
        """Set the data to visualize"""
        self.data = data
        self._create_grid()
        self._update_data_info()
        self.update_status("Data loaded successfully. Choose a visualization mode.")

        # Automatically show volume rendering
        self.render_volume()

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

    def _update_data_info(self):
        """Update data information labels"""
        if not self.data:
            return

        data_type = self.data.metadata.get('Type', 'Original')
        self.data_type_label.setText(f"Type: {data_type}")
        self.data_dim_label.setText(f"Dimensions: {self.data.dimensions}")
        self.data_spacing_label.setText(
            f"Spacing: ({self.data.spacing[0]:.2f}, "
            f"{self.data.spacing[1]:.2f}, {self.data.spacing[2]:.2f}) mm"
        )

        # Format metadata
        meta_str = "\n".join([f"{k}: {v}" for k, v in self.data.metadata.items()
                              if k != 'Type'][:3])
        self.data_meta_label.setText(f"Metadata:\n{meta_str}" if meta_str else "Metadata: None")

    def clear_view(self):
        """Clear all actors from the viewer"""
        self.plotter.clear()
        self.plotter.add_axes()
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

    def render_volume(self):
        """Render volume visualization"""
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        self.update_status("Rendering volume...")
        self.clear_view()

        # Convert cell data to point data
        vol_grid = self.grid.cell_data_to_point_data()

        # Get current colormap and opacity
        cmap = self.colormap_combo.currentText()
        opacity = self.opacity_combo.currentText()

        # Add volume rendering
        self.plotter.add_volume(vol_grid, cmap=cmap, opacity=opacity, shade=False)
        self.plotter.add_axes()
        self.plotter.reset_camera()

        self.update_status(f"Volume rendered with colormap: {cmap}, opacity: {opacity}")

    def render_slices(self):
        """Render orthogonal slices"""
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        self.update_status("Rendering orthogonal slices...")
        self.clear_view()

        slices = self.grid.slice_orthogonal()
        self.plotter.add_mesh(slices, cmap='bone')
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

        self.update_status("Orthogonal slices rendered.")

    def render_isosurface_auto(self):
        """Render isosurface with automatic threshold selection"""
        threshold = self.threshold_slider.value()
        self.render_isosurface(threshold=threshold)

    def render_isosurface(self, threshold=300, color=None, opacity=1.0):
        """Render isosurface at given threshold"""
        if not self.grid:
            QMessageBox.warning(self, "No Data", "Please load data first.")
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

        # Convert cell data to point data for contouring
        grid_points = self.grid.cell_data_to_point_data()

        try:
            contours = grid_points.contour(isosurfaces=[threshold])

            # Auto-select color based on data type if not specified
            if color is None:
                data_type = self.data.metadata.get("Type", "")
                if "Pore" in data_type or "Spheres" in data_type or "Network" in data_type:
                    color = "red"
                else:
                    color = "ivory"

            self.plotter.add_mesh(contours, color=color, opacity=opacity, smooth_shading=True)
            self.plotter.add_axes()
            self.plotter.reset_camera()

            self.update_status(f"Isosurface rendered at threshold {threshold} HU with color {color}.")
        except Exception as e:
            QMessageBox.critical(self, "Rendering Error", f"Failed to render isosurface: {str(e)}")
            self.update_status("Isosurface rendering failed.")

    # Callback methods for parameter changes

    def _on_threshold_changed(self, value):
        """Update threshold value label"""
        self.threshold_value_label.setText(f"Value: {value} HU")

    def _on_colormap_changed(self, colormap):
        """Handle colormap change - could trigger re-render if desired"""
        pass

    def _on_opacity_changed(self, opacity):
        """Handle opacity preset change - could trigger re-render if desired"""
        pass

    def closeEvent(self, event):
        """Handle window close event"""
        self.plotter.close()
        event.accept()