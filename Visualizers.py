import math
from typing import Optional, Dict, Any, List

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QMessageBox, QStatusBar, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from core import BaseVisualizer, VolumeData
from gui.panels import (
    VisualizationModePanel, 
    RenderingParametersPanel, 
    InfoPanel, 
    ClipPlanePanel, 
    ROIPanel
)


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

        # DataManager reference for centralized data flow
        self._data_manager = None

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
        from PyQt5.QtWidgets import QSplitter

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main horizontal splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Controls) - with scroll area
        left_scroll = self._create_scrollable_panel()
        self.main_splitter.addWidget(left_scroll)

        # Center Panel (3D Canvas)
        self.plotter = BackgroundPlotter(
            window_size=(1000, 900),
            show=False,
            title="3D Structure Viewer"
        )
        self.main_splitter.addWidget(self.plotter.app_window)

        # Right Panel (Info & Statistics) - with scroll area
        right_scroll = self._create_info_panel()
        self.main_splitter.addWidget(right_scroll)

        # Set initial sizes (left:center:right)
        # Using explicit sizes often yields better consistency than stretch factors alone
        self.main_splitter.setSizes([350, 900, 350])
        self.main_splitter.setCollapsible(0, False) # Prevent full collapse
        self.main_splitter.setCollapsible(2, False)

        main_layout.addWidget(self.main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")

    def _create_scrollable_panel(self) -> QWidget:
        """Create left control panel with scroll support."""
        from PyQt5.QtWidgets import QScrollArea

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Removed: setMaximumWidth - allow splitter resizing

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

        # Clip Plane Panel
        self.clip_panel = ClipPlanePanel()
        self.clip_panel.clip_toggled.connect(self._on_clip_toggled)

        # PERFORMANCE: Debounce clip updates to avoid expensive re-renders during dragging
        self.clip_update_timer = QTimer()
        self.clip_update_timer.setInterval(200)  # 200ms debounce
        self.clip_update_timer.setSingleShot(True)
        self.clip_update_timer.timeout.connect(self._apply_clip_planes)
        self.clip_panel.clip_changed.connect(lambda: self.clip_update_timer.start())

        layout.addWidget(self.clip_panel)

        # ROI Selection Panel
        self.roi_panel = ROIPanel()
        self.roi_panel.roi_toggled.connect(self._on_roi_toggled)
        self.roi_panel.apply_roi.connect(self._on_apply_roi)
        self.roi_panel.reset_roi.connect(self._on_reset_roi)
        layout.addWidget(self.roi_panel)

        layout.addStretch()

        scroll_area.setWidget(panel)
        return scroll_area

    def _create_info_panel(self) -> QWidget:
        """Create right info/statistics panel with scroll support."""
        from PyQt5.QtWidgets import QScrollArea, QSplitter

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Removed: setMaximumWidth - allow splitter resizing

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Information")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create Splitter for resizing
        self.right_splitter = QSplitter(Qt.Vertical)

        self.info_panel = InfoPanel()
        self.right_splitter.addWidget(self.info_panel)

        # Add splitter to layout
        layout.addWidget(self.right_splitter)

        scroll_area.setWidget(panel)
        return scroll_area

    def add_custom_panel(self, panel: QWidget, index: int = 2, side: str = 'left'):
        """
        Add a custom panel to either left (controls) or right (info) sidebar.

        Args:
            panel: Widget to add
            index: Position index in the layout (ignored for right side splitter)
            side: 'left' for control panel, 'right' for info panel
        """
        if side == 'left' and hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)
        elif side == 'right':
            # Add to the right splitter if available
            if hasattr(self, 'right_splitter'):
                self.right_splitter.addWidget(panel)
                # Ensure reasonable initial sizing (50/50)
                self.right_splitter.setStretchFactor(0, 1)
                self.right_splitter.setStretchFactor(1, 1)
            else:
                # Fallback to old method
                right_scroll = self.centralWidget().layout().itemAt(2).widget()
                if isinstance(right_scroll, QScrollArea):
                    right_widget = right_scroll.widget()
                    if right_widget:
                        right_widget.layout().addWidget(panel)

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
        self._iso_cache = {}  # Clear specific cache
        self._cached_vol_grid = None  # Clear volume cache

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

    def set_data_manager(self, data_manager):
        """Set reference to DataManager for centralized data flow."""
        self._data_manager = data_manager

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
            self._update_clip_panel_state()  # Update UI state
            self.params_panel.set_mode('mesh')  # Enable colormap selector
            self.plotter.enable_lightkit()

        params = self.params_panel.get_current_values()
        
        # Render with PoreRadius colormap if available
        if "PoreRadius" in self.mesh.point_data:
            # Use the PoreRadius scalar field for coloring
            self.plotter.add_mesh(
                self.mesh,
                scalars="PoreRadius",
                cmap=params['colormap'],
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Pore Radius (mm)'},
                smooth_shading=True,
                specular=0.5
            )
        elif "IsPore" in self.mesh.array_names:
            # Fallback: binary coloring for pores vs throats
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
            # Simple mesh without pore data
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
            self._update_clip_panel_state()  # Update UI state
            self.params_panel.set_mode('volume')
            self.plotter.enable_lightkit()

            # Cache the point data grid to avoid re-computation on every update
            if not hasattr(self,
                           '_cached_vol_grid') or self._cached_vol_grid is None or self._cached_vol_grid_source != id(
                    self.grid):
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
            vol_grid = self._cached_vol_grid if hasattr(self,
                                                        '_cached_vol_grid') else self.grid.cell_data_to_point_data()

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

            print(f"Updating Volume: clim={params['clim']}")  # Debug output

            if old_actor:
                self.plotter.remove_actor(old_actor, render=False)

            self.volume_actor = new_actor
            self.plotter.render()

    def _update_clip_panel_state(self):
        """Enable/Disable clip panel based on active mode."""
        if not hasattr(self, 'clip_panel'):
            return

        # Clipping supported in Volume, Mesh, and Iso modes
        supported_modes = ['volume', 'mesh', 'iso']
        is_supported = self.active_view_mode in supported_modes

        self.clip_panel.setEnabled(is_supported)

        # ALWAYS reset to disabled when switching modes (User request)
        # Block signals to prevent triggering toggle logic during reset
        self.clip_panel.enable_checkbox.blockSignals(True)
        self.clip_panel.enable_checkbox.setChecked(False)
        self.clip_panel.enable_checkbox.blockSignals(False)
        self.clip_panel._enabled = False
        self.clip_panel._update_slider_state()

    def _on_colormap_changed(self, text):
        """Handle colormap changes: Immediate for volume, Debounced for others."""
        if self.active_view_mode == 'volume':
            # Volume rendering optimization allows immediate updates
            self.render_volume(reset_view=False)
        else:
            # Other modes might require geometry processing, so debounce
            self.trigger_render()

    def _on_clip_toggled(self, enabled: bool):
        """Handle clip plane enable/disable toggle."""
        if enabled:
            self._apply_clip_planes()
        else:
            # CRITICAL: Force full re-render to restore original unclipped data
            # Using reset_view=True ensures the cached/clipped data is discarded
            if self.active_view_mode == 'volume':
                self.render_volume(reset_view=True)
            elif self.active_view_mode == 'slices':
                self.render_slices(reset_view=True)
            elif self.active_view_mode == 'iso':
                self.render_isosurface_auto(reset_view=True)
            elif self.active_view_mode == 'mesh':
                self.render_mesh(reset_view=True)

    def _apply_clip_planes(self):
        """Apply clip planes to current visualization based on panel settings."""
        if not hasattr(self, 'clip_panel'):
            return

        clip_vals = self.clip_panel.get_clip_values()
        if not clip_vals['enabled']:
            return

        # CLAMP: Ensure minimum clip thickness to prevent empty mesh errors
        # User requested minimum size instead of 0
        EPS = 0.005  # 0.5%
        for axis in ['x', 'y', 'z']:
            # If standard (0->1), clamp min to EPS
            # If invert (1->0), clamp max to 1-EPS
            if not clip_vals[f'invert_{axis}']:
                clip_vals[axis] = max(EPS, clip_vals[axis])
            else:
                clip_vals[axis] = min(1.0 - EPS, clip_vals[axis])

        try:
            # Determine rendering bounds and data source based on MODE
            data_source = None
            if self.active_view_mode == 'volume':
                data_source = self.grid
            elif self.active_view_mode == 'mesh':
                data_source = self.mesh
            elif self.active_view_mode == 'iso':
                # For ISO, we need the *current* isosurface
                # We can get it from cache using current threshold
                params = self.params_panel.get_current_values()
                thresh = params['threshold']
                if thresh in self._iso_cache:
                    data_source = self._iso_cache[thresh]
                elif self.grid:
                    # Fallback: generating on the fly (should generally exist from previous render)
                    data_source = self.grid.cell_data_to_point_data().contour([thresh])

            if data_source is None:
                return

            bounds = data_source.bounds

            # Calculate clip box ...
            x_min = bounds[0]
            x_max = bounds[0] + (bounds[1] - bounds[0]) * clip_vals['x']
            y_min = bounds[2]
            y_max = bounds[2] + (bounds[3] - bounds[2]) * clip_vals['y']
            z_min = bounds[4]
            z_max = bounds[4] + (bounds[5] - bounds[4]) * clip_vals['z']

            # Handle invert
            if clip_vals['invert_x']: x_min, x_max = x_max, bounds[1]
            if clip_vals['invert_y']: y_min, y_max = y_max, bounds[3]
            if clip_vals['invert_z']: z_min, z_max = z_max, bounds[5]

            clip_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

            # --- RENDER ---
            self.plotter.clear()
            self.plotter.add_axes()
            params = self.params_panel.get_current_values()

            # Mode 1: Volume
            if self.active_view_mode == 'volume' and self.grid is not None:
                # Use clip_box on the GRID
                clipped = self.grid.clip_box(clip_bounds, invert=False)
                if clipped.n_cells > 0:
                    # Render clipped volume as "Mesh" (Isosurface-like) to avoid VolumeMapper crash
                    # OR if we want real volume rendering, we must use add_volume.
                    # Previous crash issue: add_volume doesn't like UnstructuredGrid from clip().
                    # Solution: Smart volume rendering or ExtractRegion?
                    # For stability: Render as semi-transparent surface representing the volume data
                    self.plotter.add_mesh(
                        clipped,
                        scalars="values",
                        cmap=params['colormap'],
                        clim=params['clim'],
                        show_scalar_bar=True,
                        opacity=0.5  # Semi-transparent to look like volume
                    )

            # Mode 2: Mesh (PNM)
            elif self.active_view_mode == 'mesh':
                clipped = self.mesh.clip_box(clip_bounds, invert=False)
                if clipped.n_points > 0:
                    self.plotter.add_mesh(
                        clipped,
                        scalars="IsPore" if "IsPore" in clipped.array_names else None,
                        cmap=["gray", "red"] if "IsPore" in clipped.array_names else params['colormap'],
                        show_scalar_bar=False,
                        smooth_shading=True
                    )

            # Mode 3: Isosurface
            elif self.active_view_mode == 'iso':
                # Clip the PolyData surface
                clipped = data_source.clip_box(clip_bounds, invert=False)

                # Re-apply iso styles
                style_map = {'Surface': 'surface', 'Wireframe': 'wireframe', 'Wireframe + Surface': 'surface'}
                render_style = style_map.get(params['render_style'], 'surface')
                show_edges = (params['render_style'] == 'Wireframe + Surface')

                mesh_kwargs = {
                    'style': render_style, 'show_edges': show_edges, 'smooth_shading': True,
                    'specular': 0.4, 'diffuse': 0.7, 'ambient': 0.15, 'lighting': True
                }

                mode = params['coloring_mode']
                if mode == 'Solid Color':
                    self.plotter.add_mesh(clipped, color=params['solid_color'], **mesh_kwargs)
                elif mode == 'Depth (Z-Axis)':
                    clipped["Elevation"] = clipped.points[:, 2]
                    self.plotter.add_mesh(clipped, scalars="Elevation", cmap=params['colormap'], **mesh_kwargs)
                else:
                    self.plotter.add_mesh(clipped, color='white', **mesh_kwargs)

            # Re-apply lighting settings (critical after plotter.clear())
            self._apply_custom_lighting(params)

            self.plotter.render()

        except Exception as e:
            print(f"[Clip] Error: {e}")
            self._on_clip_toggled(False)

    # ==========================================
    # ROI Selection Methods
    # ==========================================

    def _on_roi_toggled(self, enabled: bool):
        """Handle ROI mode toggle - show/hide box widget."""
        if enabled:
            if self.grid is None:
                self.roi_panel.enable_checkbox.setChecked(False)
                return

            # Add interactive box widget
            bounds = self.grid.bounds
            self.plotter.add_box_widget(
                callback=self._on_roi_bounds_changed,
                bounds=bounds,
                factor=1.0,
                rotation_enabled=False,
                color='cyan',
                use_planes=False
            )
            self.update_status("ROI mode: Drag the box to select region")
        else:
            # Remove box widget
            self.plotter.clear_box_widgets()
            self.roi_panel.update_bounds(None)
            self.update_status("ROI mode disabled")

    def _on_roi_bounds_changed(self, bounds):
        """Callback when user moves the ROI box widget."""
        # bounds is a PolyData box, extract actual bounds
        if hasattr(bounds, 'bounds'):
            actual_bounds = bounds.bounds
        else:
            actual_bounds = bounds
        self.roi_panel.update_bounds(actual_bounds)

    def _on_apply_roi(self):
        """Extract sub-volume from ROI bounds and update data."""
        roi_bounds = self.roi_panel.get_bounds()
        if roi_bounds is None or self.data is None:
            return

        try:
            # Extract sub-volume from the grid
            extracted = self._extract_roi_subvolume(roi_bounds)
            if extracted is not None:
                # Store in DataManager for centralized access
                if self._data_manager is not None:
                    self._data_manager.set_roi_data(extracted)

                # Update visualization with extracted data
                self.set_data(extracted)
                self.update_status(f"ROI applied: {extracted.raw_data.shape}")

                # Disable ROI mode after applying
                self.roi_panel.enable_checkbox.setChecked(False)
                self.plotter.clear_box_widgets()
        except Exception as e:
            print(f"[ROI] Error applying: {e}")
            self.update_status("ROI extraction failed")

    def _on_reset_roi(self):
        """Reset ROI - clear box widget and reset to original data."""
        self.plotter.clear_box_widgets()

        # Clear ROI in DataManager and restore to raw data
        if self._data_manager is not None:
            self._data_manager.clear_roi()
            if self._data_manager.raw_ct_data is not None:
                self.set_data(self._data_manager.raw_ct_data)

        self.update_status("ROI reset")

    def _extract_roi_subvolume(self, bounds) -> Optional[VolumeData]:
        """Extract a sub-volume based on ROI bounds."""
        if self.data is None or self.data.raw_data is None:
            return None

        # Convert physical bounds to voxel indices
        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin

        # AXIS MAPPING FIX:
        # PyVista ImageData was created with: grid.dimensions = shape (which is Z, Y, X in numpy)
        # So PyVista's X axis corresponds to numpy axis 0 (Z dimension)
        # PyVista's Y axis corresponds to numpy axis 1 (Y dimension)
        # PyVista's Z axis corresponds to numpy axis 2 (X dimension)
        #
        # bounds from PyVista = (xmin, xmax, ymin, ymax, zmin, zmax)
        # We need to map: PyVista X -> numpy axis 0, PyVista Y -> numpy axis 1, PyVista Z -> numpy axis 2

        # Get numpy indices for each PyVista axis
        # PyVista X (bounds 0,1) -> numpy axis 0
        i_start = max(0, int((bounds[0] - origin[0]) / spacing[0]))
        i_end = min(raw.shape[0], int((bounds[1] - origin[0]) / spacing[0]))

        # PyVista Y (bounds 2,3) -> numpy axis 1
        j_start = max(0, int((bounds[2] - origin[1]) / spacing[1]))
        j_end = min(raw.shape[1], int((bounds[3] - origin[1]) / spacing[1]))

        # PyVista Z (bounds 4,5) -> numpy axis 2
        k_start = max(0, int((bounds[4] - origin[2]) / spacing[2]))
        k_end = min(raw.shape[2], int((bounds[5] - origin[2]) / spacing[2]))

        # Extract sub-array using corrected axis order
        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end]

        if sub_data.size == 0:
            return None

        # Create new VolumeData with extracted region
        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )

        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Extract ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds

        return VolumeData(
            raw_data=sub_data,
            spacing=spacing,
            origin=new_origin,
            metadata=new_metadata
        )

    def render_slices(self, reset_view=True):
        if not self.grid: return

        if reset_view or self.active_view_mode != 'slices':
            self.clear_view()
            self.active_view_mode = 'slices'
            self._update_clip_panel_state()  # Update UI state
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
        self._update_clip_panel_state()  # Update UI state
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

            # Apply lighting
            self._apply_custom_lighting(params)

            self.plotter.add_axes()
            if reset_view: self.reset_camera()
        except Exception as e:
            print(e)

    def _apply_custom_lighting(self, params):
        """Apply custom lighting configuration based on parameters."""
        if 'light_angle' in params and params['light_angle'] is not None:
            angle = params['light_angle']
            # Light position: convert angle to 3D position
            import math
            rad = math.radians(angle)
            # Position light in a circle around Z axis
            light_pos = [10 * math.cos(rad), 10 * math.sin(rad), 10]
            self.plotter.remove_all_lights()
            self.plotter.add_light(pv.Light(position=light_pos, intensity=1.0))
            self.update_status("Error generating isosurface.")


# ==========================================
# Renderer Class (Decoupled from QMainWindow)
# ==========================================

class Renderer:
    """
    Pure 3D rendering class.
    Handles PyVista rendering without any QMainWindow dependencies.
    Receives panel references and plotter from MainWindow.
    """
    
    def __init__(self, plotter, params_panel, info_panel, clip_panel, roi_panel, status_callback=None):
        """
        Initialize renderer with external dependencies.
        
        Args:
            plotter: BackgroundPlotter instance
            params_panel: RenderingParametersPanel instance
            info_panel: InfoPanel instance
            clip_panel: ClipPlanePanel instance
            roi_panel: ROIPanel instance
            status_callback: Function to call for status updates
        """
        self.plotter = plotter
        self.params_panel = params_panel
        self.info_panel = info_panel
        self.clip_panel = clip_panel
        self.roi_panel = roi_panel
        self._status_callback = status_callback
        
        # Data state
        self.data: Optional[VolumeData] = None
        self.grid: Optional[pv.ImageData] = None
        self.mesh: Optional[pv.PolyData] = None
        self.active_view_mode: Optional[str] = None
        
        # Caches
        self._iso_cache: Dict[int, pv.PolyData] = {}
        self._cached_vol_grid: Optional[pv.PolyData] = None
        self._cached_vol_grid_source: Optional[int] = None
        
        # Actors
        self.volume_actor = None
        
        # Data manager reference
        self._data_manager = None
    
    def update_status(self, message: str):
        """Update status via callback."""
        if self._status_callback:
            self._status_callback(message)
        else:
            print(f"[Renderer] {message}")
    
    def set_data_manager(self, data_manager):
        """Set reference to DataManager for centralized data flow."""
        self._data_manager = data_manager
    
    def set_data(self, data: VolumeData):
        """Load data and trigger appropriate rendering."""
        self.data = data
        self.grid = None
        self.mesh = None
        self.volume_actor = None
        self._iso_cache = {}
        self._cached_vol_grid = None
        
        d_type = self.data.metadata.get('Type', 'Unknown')
        
        if self.data.has_mesh:
            self.mesh = self.data.mesh
            self.update_status(f"Loaded Mesh: {d_type}")
            self.render_mesh(reset_view=True)
            self.info_panel.update_info(d_type, (0, 0, 0), self.data.spacing, self.data.metadata)
        
        elif self.data.raw_data is not None:
            self._create_pyvista_grid()
            self.update_status(f"Loaded Volume: {d_type}")
            
            is_processed = ("Processed" in d_type)
            default_color = "red" if is_processed else "ivory"
            idx = self.params_panel.solid_color_combo.findText(default_color)
            if idx >= 0:
                self.params_panel.solid_color_combo.setCurrentIndex(idx)
            
            min_val = np.nanmin(self.data.raw_data)
            max_val = np.nanmax(self.data.raw_data)
            self.params_panel.set_data_range(min_val, max_val)
            
            dims = self.grid.dimensions
            self.params_panel.set_slice_limits(dims[0] - 1, dims[1] - 1, dims[2] - 1)
            self.params_panel.set_slice_defaults(dims[0] // 2, dims[1] // 2, dims[2] // 2)
            
            self.render_volume(reset_view=True)
            self.info_panel.update_info(d_type, self.data.dimensions, self.data.spacing, self.data.metadata)
    
    def _create_pyvista_grid(self):
        if not self.data or self.data.raw_data is None:
            return
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
        """Render PNM mesh with PoreRadius colormap."""
        if not self.mesh:
            return
        if reset_view or self.active_view_mode != 'mesh':
            self.clear_view()
            self.active_view_mode = 'mesh'
            self._update_clip_panel_state()
            self.params_panel.set_mode('mesh')
            self.plotter.enable_lightkit()
        
        params = self.params_panel.get_current_values()
        
        if "PoreRadius" in self.mesh.point_data:
            self.plotter.add_mesh(
                self.mesh,
                scalars="PoreRadius",
                cmap=params['colormap'],
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Pore Radius (mm)'},
                smooth_shading=True,
                specular=0.5
            )
        elif "IsPore" in self.mesh.array_names:
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
        
        if reset_view:
            self.reset_camera()
    
    def render_volume(self, reset_view=True):
        """Optimized volume rendering."""
        if not self.grid:
            return
        
        if reset_view or self.active_view_mode != 'volume' or self.volume_actor is None:
            self.update_status("Rendering volume (New)...")
            self.clear_view()
            self.active_view_mode = 'volume'
            self._update_clip_panel_state()
            self.params_panel.set_mode('volume')
            self.plotter.enable_lightkit()
            
            if not hasattr(self, '_cached_vol_grid') or self._cached_vol_grid is None or self._cached_vol_grid_source != id(self.grid):
                self._cached_vol_grid = self.grid.cell_data_to_point_data()
                self._cached_vol_grid_source = id(self.grid)
            
            vol_grid = self._cached_vol_grid
            params = self.params_panel.get_current_values()
            
            self.volume_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params['colormap'],
                opacity=params['opacity'],
                clim=params['clim'],
                shade=False
            )
            self.plotter.add_axes()
            if reset_view:
                self.reset_camera()
        else:
            self.update_status("Updating volume properties...")
            params = self.params_panel.get_current_values()
            vol_grid = self._cached_vol_grid if hasattr(self, '_cached_vol_grid') else self.grid.cell_data_to_point_data()
            
            old_actor = self.volume_actor
            new_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params['colormap'],
                opacity=params['opacity'],
                clim=params['clim'],
                shade=False,
                render=False
            )
            
            if new_actor.mapper:
                new_actor.mapper.scalar_range = params['clim']
            
            if old_actor:
                self.plotter.remove_actor(old_actor, render=False)
            
            self.volume_actor = new_actor
            self.plotter.render()
    
    def render_slices(self, reset_view=True):
        """Render orthogonal slices."""
        if not self.grid:
            return
        
        if reset_view or self.active_view_mode != 'slices':
            self.clear_view()
            self.active_view_mode = 'slices'
            self._update_clip_panel_state()
            self.params_panel.set_mode('slices')
            self.plotter.enable_lightkit()
            self.plotter.show_grid()
        
        params = self.params_panel.get_current_values()
        ox, oy, oz = self.grid.origin
        dx, dy, dz = self.grid.spacing
        x = ox + params['slice_x'] * dx
        y = oy + params['slice_y'] * dy
        z = oz + params['slice_z'] * dz
        
        self.plotter.clear_actors()
        slices = self.grid.slice_orthogonal(x=x, y=y, z=z)
        self.plotter.add_mesh(slices, cmap=params['colormap'], clim=params['clim'], show_scalar_bar=False)
        self.plotter.add_axes()
        if reset_view:
            self.reset_camera()
    
    def render_isosurface_auto(self, reset_view=True):
        """Render isosurface with current threshold."""
        if not self.grid:
            return
        params = self.params_panel.get_current_values()
        self.render_isosurface(threshold=params['threshold'], reset_view=reset_view)
    
    def render_isosurface(self, threshold=300, reset_view=True):
        """Render isosurface at specified threshold."""
        if not self.grid:
            return
        
        self.update_status(f"Generating isosurface ({threshold})...")
        
        if reset_view or self.active_view_mode != 'iso':
            self.clear_view()
            self.active_view_mode = 'iso'
            self._update_clip_panel_state()
            self.params_panel.set_mode('iso')
            self.plotter.enable_lightkit()
        
        try:
            if threshold in self._iso_cache:
                iso_mesh = self._iso_cache[threshold]
            else:
                iso_mesh = self.grid.cell_data_to_point_data().contour([threshold])
                self._iso_cache[threshold] = iso_mesh
            
            if iso_mesh.n_points == 0:
                self.update_status("No surface at this threshold.")
                return
            
            params = self.params_panel.get_current_values()
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
                self.plotter.add_mesh(iso_mesh, color=params['solid_color'], **mesh_kwargs)
            elif mode == 'Depth (Z-Axis)':
                iso_mesh["Elevation"] = iso_mesh.points[:, 2]
                self.plotter.add_mesh(iso_mesh, scalars="Elevation", cmap=params['colormap'], **mesh_kwargs)
            elif mode == 'Radial (Center Dist)':
                center = iso_mesh.center
                distances = np.linalg.norm(iso_mesh.points - center, axis=1)
                iso_mesh["Radial"] = distances
                self.plotter.add_mesh(iso_mesh, scalars="Radial", cmap=params['colormap'], **mesh_kwargs)
            
            self._apply_custom_lighting(params)
            
            if reset_view:
                self.reset_camera()
        except Exception as e:
            print(e)
            self.update_status("Error generating isosurface.")
    
    def _update_clip_panel_state(self):
        """Enable/Disable clip panel based on active mode."""
        if not self.clip_panel:
            return
        
        supported_modes = ['volume', 'mesh', 'iso']
        is_supported = self.active_view_mode in supported_modes
        self.clip_panel.setEnabled(is_supported)
        
        self.clip_panel.enable_checkbox.blockSignals(True)
        self.clip_panel.enable_checkbox.setChecked(False)
        self.clip_panel.enable_checkbox.blockSignals(False)
        self.clip_panel._enabled = False
        self.clip_panel._update_slider_state()
    
    def _on_clip_toggled(self, enabled: bool):
        """Handle clip plane toggle."""
        if enabled:
            self._apply_clip_planes()
        else:
            if self.active_view_mode == 'volume':
                self.render_volume(reset_view=True)
            elif self.active_view_mode == 'slices':
                self.render_slices(reset_view=True)
            elif self.active_view_mode == 'iso':
                self.render_isosurface_auto(reset_view=True)
            elif self.active_view_mode == 'mesh':
                self.render_mesh(reset_view=True)
    
    def _apply_clip_planes(self):
        """Apply clip planes (simplified version)."""
        # Implementation would be similar to GuiVisualizer._apply_clip_planes
        pass
    
    def _on_roi_toggled(self, enabled: bool):
        """Handle ROI mode toggle."""
        if enabled and self.grid:
            bounds = self.grid.bounds
            self.plotter.add_box_widget(
                callback=self._on_roi_bounds_changed,
                bounds=bounds,
                factor=1.0,
                rotation_enabled=False,
                color='cyan',
                use_planes=False
            )
            self.update_status("ROI mode: Drag the box to select region")
        else:
            self.plotter.clear_box_widgets()
            self.roi_panel.update_bounds(None)
            self.update_status("ROI mode disabled")
    
    def _on_roi_bounds_changed(self, bounds):
        """Callback when ROI box moves."""
        if hasattr(bounds, 'bounds'):
            actual_bounds = bounds.bounds
        else:
            actual_bounds = bounds
        self.roi_panel.update_bounds(actual_bounds)
    
    def _on_apply_roi(self):
        """Apply ROI selection."""
        # Implementation similar to GuiVisualizer._on_apply_roi
        pass
    
    def _on_reset_roi(self):
        """Reset ROI selection."""
        self.plotter.clear_box_widgets()
        if self._data_manager:
            self._data_manager.clear_roi()
            if self._data_manager.raw_ct_data:
                self.set_data(self._data_manager.raw_ct_data)
        self.update_status("ROI reset")
    
    def _apply_custom_lighting(self, params):
        """Apply custom lighting configuration."""
        if 'light_angle' in params and params['light_angle'] is not None:
            angle = params['light_angle']
            rad = math.radians(angle)
            light_pos = [10 * math.cos(rad), 10 * math.sin(rad), 10]
            self.plotter.remove_all_lights()
            self.plotter.add_light(pv.Light(position=light_pos, intensity=1.0))