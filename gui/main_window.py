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
from rendering.roi_handler import ROIHandler
from rendering.clip_handler import ClipHandler


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

        self.setWindowTitle("Porous CT Analysis Suite")
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
        
        # Initialize ROIHandler
        self.roi_handler = ROIHandler(
            plotter=self.plotter,
            roi_panel=self.roi_panel,
            data_manager=self._data_manager,
            status_callback=self.update_status
        )
        
        # Initialize ClipHandler
        self.clip_handler = ClipHandler(
            plotter=self.plotter,
            clip_panel=self.clip_panel,
            render_engine=self.render_engine
        )
        
        # Connect signals to handlers
        self._connect_roi_signals()
        self._connect_clip_signals()

        # Debounce timer for expensive renders
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_delayed_render)

        self._setup_mouse_probe()
    
    def _connect_roi_signals(self):
        """Connect ROI panel signals to ROIHandler."""
        self.roi_panel.roi_toggled.connect(self.roi_handler.on_toggled)
        self.roi_panel.apply_roi.connect(lambda: self.roi_handler.on_apply(self.set_data))
        self.roi_panel.reset_roi.connect(lambda: self.roi_handler.on_reset(self.set_data))
        self.roi_panel.shape_changed.connect(self.roi_handler.on_shape_changed)
    
    def _connect_clip_signals(self):
        """Connect Clip panel signals to ClipHandler."""
        self.clip_panel.clip_toggled.connect(self.clip_handler.on_clip_toggled)
        self.clip_update_timer = QTimer()
        self.clip_update_timer.setInterval(200)
        self.clip_update_timer.setSingleShot(True)
        self.clip_update_timer.timeout.connect(self.clip_handler.apply_clip_planes)
        self.clip_panel.clip_changed.connect(lambda: self.clip_update_timer.start())

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

        # Clip Panel - signals connected later in __init__ after clip_handler is created
        self.clip_panel = ClipPlanePanel()
        layout.addWidget(self.clip_panel)

        # ROI Panel - signals connected later in __init__ after roi_handler is created
        self.roi_panel = ROIPanel()
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
        
        # Sync with ROIHandler
        if hasattr(self, 'roi_handler'):
            self.roi_handler.set_data(data, self.render_engine.grid)
        
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
            
            # Update Histogram
            self._update_histogram()

            min_val = np.nanmin(data.raw_data)
            max_val = np.nanmax(data.raw_data)
            self.params_panel.set_data_range(min_val, max_val)

            dims = self.render_engine.grid.dimensions
            self.params_panel.set_slice_limits(dims[0] - 1, dims[1] - 1, dims[2] - 1)
            self.params_panel.set_slice_defaults(dims[0] // 2, dims[1] // 2, dims[2] // 2)

            self.render_volume(reset_view=True)
            self.info_panel.update_info(d_type, data.dimensions, data.spacing, data.metadata)

    def _update_histogram(self):
        """Calculate and update histogram in params panel using DataManager."""
        if not self._data_manager:
            return
            
        try:
            hist, bins = self._data_manager.calculate_histogram(bins=100, sample_step=4)
            if hist.size > 0:
                self.params_panel.set_histogram_data(hist, bins)
        except Exception as e:
            print(f"Histogram error: {e}")

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
        # Update ROIHandler's data manager reference
        if hasattr(self, 'roi_handler'):
            self.roi_handler._data_manager = data_manager

    # ==========================================
    # Helper Methods for Data Operations
    # ==========================================
    
    def _clear_render_caches(self):
        """Clear all render engine caches. Called before data modification."""
        import gc
        self.plotter.clear()
        if hasattr(self, 'render_engine'):
            self.render_engine.data = None
            self.render_engine.grid = None
            self.render_engine._cached_vol_grid = None
            self.render_engine._lod_pyramid = None
            self.render_engine._iso_cache = {}
            self.render_engine.volume_actor = None
        gc.collect()
    
    def _refresh_view(self):
        """Re-render current view mode."""
        if self.active_view_mode == 'volume':
            self.render_volume(reset_view=True)
        elif self.active_view_mode == 'slices':
            self.render_slices(reset_view=True)
        elif self.active_view_mode == 'iso':
            self.render_isosurface_auto(reset_view=True)
        elif self.active_view_mode == 'mesh':
            self.render_mesh(reset_view=True)
        else:
            self.render_volume(reset_view=True)

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
        """Apply colormap range as permanent data clip (delegates to DataManager)."""
        if not hasattr(self, '_data_manager') or not self._data_manager:
            self.update_status("No data to clip")
            return
        
        import gc
        import traceback
        
        min_val, max_val = float(clim[0]), float(clim[1])
        
        try:
            # Step 1: Clear render caches
            self.update_status("Clipping: Clearing caches...")
            self._clear_render_caches()
            
            # Step 2: Delegate data manipulation to DataManager
            def progress_cb(percent, msg):
                self.update_status(f"Clipping: {msg}")
            
            self._data_manager.clip_data(min_val, max_val, progress_callback=progress_cb)
            
            # Step 3: Update UI
            self.update_status("Clipping: Updating UI...")
            self.params_panel.set_data_range(int(min_val), int(max_val))
            
            # Step 4: Recreate rendering grid
            self.update_status("Clipping: Recreating grid...")
            self.render_engine.set_data(self._data_manager.active_data)
            
            # Step 5: Update histogram
            self._update_histogram()
            
            # Step 6: Re-render
            self._refresh_view()
            
            self.update_status(f"Applied clip: [{min_val:.0f}, {max_val:.0f}]")
            gc.collect()
            
        except ValueError as e:
            self.update_status(str(e))
        except MemoryError as e:
            print(f"[RangeClip] Memory Error: {e}")
            print(traceback.format_exc())
            self.update_status("Memory Error during clip")
            gc.collect()
        except Exception as e:
            print(f"[RangeClip] Error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.update_status(f"Clip error: {type(e).__name__}")

    def _on_invert_volume(self):
        """Invert volume values (delegates to DataManager)."""
        if not hasattr(self, '_data_manager') or not self._data_manager:
            self.update_status("No data to invert")
            return
        
        import gc
        import traceback
        
        try:
            # Step 1: Clear render caches
            self.update_status("Inverting: Clearing caches...")
            self._clear_render_caches()
            
            # Step 2: Get current clim before inversion (for UI update)
            data = self._data_manager.active_data
            current_clim = self.params_panel.get_current_values().get('clim', [0, 1000])
            old_min_clim, old_max_clim = current_clim[0], current_clim[1]
            
            # Step 3: Delegate data inversion to DataManager
            def progress_cb(percent, msg):
                self.update_status(f"Inverting: {msg}")
            
            data_min, data_max, invert_offset = self._data_manager.invert_data(progress_callback=progress_cb)
            
            # Step 4: Calculate inverted clim values for UI
            new_min_clim = int(invert_offset - old_max_clim)
            new_max_clim = int(invert_offset - old_min_clim)
            
            # Step 5: Update params panel
            self.update_status("Inverting: Updating UI...")
            self.params_panel.set_data_range(int(data_min), int(data_max))
            self.params_panel.block_signals(True)
            self.params_panel.slider_clim_min.setValue(new_min_clim)
            self.params_panel.slider_clim_max.setValue(new_max_clim)
            self.params_panel.spinbox_clim_min.setValue(new_min_clim)
            self.params_panel.spinbox_clim_max.setValue(new_max_clim)
            self.params_panel.block_signals(False)
            
            # Step 6: Recreate rendering grid
            self.update_status("Inverting: Recreating grid...")
            self.render_engine.set_data(self._data_manager.active_data)
            
            # Step 7: Update histogram
            self._update_histogram()
            
            # Step 8: Re-render
            self._refresh_view()
            
            self.update_status(f"Volume inverted. Clim: [{new_min_clim}, {new_max_clim}]")
            gc.collect()
            
        except ValueError as e:
            self.update_status(str(e))
        except Exception as e:
            print(f"[InvertVolume] Error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.update_status(f"Invert error: {type(e).__name__}")


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
