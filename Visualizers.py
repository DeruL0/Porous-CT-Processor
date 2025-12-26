import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QStatusBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from typing import Optional

from Core import BaseVisualizer, VolumeData
# Import decoupled UI components
from GUI import InfoPanel, AnalysisModePanel, RenderingParamsPanel, Separator


# Fix for metaclass conflict
class VisualizerMeta(type(QMainWindow), type(BaseVisualizer)):
    pass


class GuiVisualizer(QMainWindow, BaseVisualizer, metaclass=VisualizerMeta):
    """
    PyQt5-based GUI Visualizer.
    Responsible for:
    1. Creating the Main Window layout.
    2. Embedding the PyVista Plotter.
    3. Updating the 3D scene based on Data or User Events.

    It is NOT responsible for defining Application Logic buttons (Loading/Processing).
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[VolumeData] = None
        self.grid = None

        # Window setup
        self.setWindowTitle("Porous Media Analysis Suite (Micro-CT)")
        self.setGeometry(100, 100, 1400, 900)

        # UI Components References
        self.info_panel = None
        self.mode_panel = None
        self.params_panel = None
        self.control_layout = None

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface structure"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Control Panel Container
        control_container = QWidget()
        control_container.setMaximumWidth(400)
        self.control_layout = QVBoxLayout(control_container)
        self.control_layout.setSpacing(10)

        # Title
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(title)
        self.control_layout.addWidget(Separator())

        # 1. Info Panel
        self.info_panel = InfoPanel()
        self.control_layout.addWidget(self.info_panel)

        # 2. Analysis Mode Panel (Visualization logic)
        self.mode_panel = AnalysisModePanel()
        self.mode_panel.mode_changed.connect(self._handle_mode_change)
        self.control_layout.addWidget(self.mode_panel)

        # 3. Rendering Parameters Panel
        self.params_panel = RenderingParamsPanel()
        self.params_panel.threshold_changed.connect(self._on_threshold_changed)
        # Note: We don't auto-re-render on color/opacity change to save perf,
        # but in a real app we might.
        self.control_layout.addWidget(self.params_panel)

        # Add stretch at the end to push widgets up
        self.control_layout.addStretch()

        main_layout.addWidget(control_container, stretch=1)

        # Right: 3D Viewer
        self.plotter = BackgroundPlotter(window_size=(1000, 900), show=False, title="3D Structure Viewer")
        main_layout.addWidget(self.plotter.app_window, stretch=3)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")

    def add_custom_panel(self, panel_widget: QWidget, index: int = 2):
        """
        Allows the Controller to inject a custom panel (e.g. Workflow/Processing buttons)
        into the control layout without accessing the layout directly.
        """
        # Insert before the stretch (which is the last item)
        # Or at a specific index.
        # Index 0=Title, 1=Separator, 2=Info, 3=Mode, 4=Params.
        # We usually want logic buttons near the top or middle.
        self.control_layout.insertWidget(index, panel_widget)

    def set_data(self, data: VolumeData):
        """Set the data and update Info Panel"""
        self.data = data
        self._create_grid()

        # Update UI Panel
        self.info_panel.update_info(
            type_str=data.metadata.get('Type', 'Original Scan'),
            dim=data.dimensions,
            spacing=data.spacing,
            meta_dict=data.metadata
        )

        self.update_status("Scan loaded successfully.")
        self.render_volume()

    def _create_grid(self):
        """Create PyVista grid from volume data"""
        if not self.data: return
        grid = pv.ImageData()
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    # ==========================================
    # Visualization Logic (View)
    # ==========================================

    def _handle_mode_change(self, mode):
        """Dispatcher for visualization modes"""
        if mode == "volume":
            self.render_volume()
        elif mode == "slices":
            self.render_slices()
        elif mode == "iso":
            self.render_isosurface_auto()
        elif mode == "clear":
            self.clear_view()
        elif mode == "reset":
            self.reset_camera()

    def render_volume(self):
        if not self.grid: return self._alert_no_data()
        self.update_status("Rendering volume...")
        self.clear_view()

        vol_grid = self.grid.cell_data_to_point_data()
        cmap = self.params_panel.get_current_colormap()
        opacity = self.params_panel.get_current_opacity()

        self.plotter.add_volume(vol_grid, cmap=cmap, opacity=opacity, shade=False)
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.update_status(f"Volume rendered ({cmap}).")

    def render_slices(self):
        if not self.grid: return self._alert_no_data()
        self.update_status("Rendering slices...")
        self.clear_view()

        slices = self.grid.slice_orthogonal()
        self.plotter.add_mesh(slices, cmap='bone')
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()
        self.update_status("Orthogonal slices rendered.")

    def render_isosurface_auto(self):
        val = self.params_panel.get_current_threshold()
        self.render_isosurface(val)

    def render_isosurface(self, threshold):
        if not self.grid: return self._alert_no_data()
        self.update_status(f"Extracting isosurface at {threshold}...")
        self.clear_view()

        grid_points = self.grid.cell_data_to_point_data()
        try:
            contours = grid_points.contour(isosurfaces=[threshold])

            # Smart color selection
            color = "red" if "Pore" in self.data.metadata.get("Type", "") else "ivory"

            self.plotter.add_mesh(contours, color=color, opacity=1.0, smooth_shading=True)
            self.plotter.add_axes()
            self.plotter.reset_camera()
            self.update_status(f"Isosurface at {threshold} rendered.")
        except Exception as e:
            self.update_status("Rendering failed.")
            print(f"Error: {e}")

    def clear_view(self):
        self.plotter.clear()
        self.plotter.add_axes()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def update_status(self, msg):
        self.status_bar.showMessage(msg)

    def _alert_no_data(self):
        QMessageBox.warning(self, "No Data", "Please load a sample first.")

    def _on_threshold_changed(self, val):
        # We don't auto-render isosurfaces as it is heavy, just update text is handled in UI class
        pass

    def show(self):
        super().show()
        self.plotter.app_window.show()