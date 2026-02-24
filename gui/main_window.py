"""
Main application window combining PyQt5 UI with PyVista 3D rendering.
Delegates rendering logic to RenderEngine for separation of concerns.
"""

import math
import weakref
from typing import Any, Dict, List, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QFrame, QStatusBar, QScrollArea, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QSignalBlocker
from PyQt5.QtGui import QFont, QCloseEvent

from core import VolumeData
from core.coordinates import world_xyz_to_index_zyx
from gui.panels import (
    VisualizationModePanel,
    RenderingParametersPanel,
    InfoPanel,
    ClipPlanePanel,
    ROIPanel
)
from gui.ui_constants import PANEL_MARGIN, PANEL_SPACING
from gui.widgets import CollapsiblePanel
from rendering import RenderEngine
from rendering.roi_handler import ROIHandler
from rendering.clip_handler import ClipHandler


class MainWindow(QMainWindow):
    """
    Main application window.
    Integrates PyQt5 UI controls with a PyVista 3D rendering canvas.
    Delegates rendering to RenderEngine for separation of concerns.

    Note: previously inherited from both QMainWindow and BaseVisualizer (ABC),
    which required a _MainWindowMeta metaclass hack.  BaseVisualizer is now a
    typing.Protocol so no metaclass resolution is needed.
    """

    def __init__(self):
        super().__init__()
        self._data_manager = None
        self._active_overlay_layers: List[Dict[str, Any]] = []
        self._replaying_overlays = False
        self._timeseries_handler = None

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
        self._sync_render_style_ui()
        
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
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setContentsMargins(*PANEL_MARGIN)
        layout.setSpacing(PANEL_SPACING)

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
        self.mode_panel.overlay_layer_requested.connect(self._on_add_overlay_layer)
        self.mode_panel.overlays_cleared.connect(self._on_clear_overlays)
        layout.addWidget(self.mode_panel)

        # Parameters Panel
        self.params_panel = RenderingParametersPanel()
        for signal in [self.params_panel.solid_color_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.threshold_changed,
                       self.params_panel.slice_position_changed]:
            signal.connect(self.trigger_render)
        self.params_panel.render_style_changed.connect(self._on_render_style_changed)

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

        layout.addStretch(1)
        scroll_area.setWidget(panel)
        return scroll_area

    def _create_info_panel(self) -> QWidget:
        """Create right sidebar with global scroll area."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        panel = QWidget()
        self.info_panel_layout = QVBoxLayout(panel)
        layout = self.info_panel_layout
        layout.setContentsMargins(*PANEL_MARGIN)
        layout.setSpacing(PANEL_SPACING)
        
        # Title
        title = QLabel("Information")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())
        
        # Default Info Panel
        self.info_panel = InfoPanel()
        info_wrapper = CollapsiblePanel(
            title=self._resolve_panel_title(self.info_panel),
            content=self.info_panel,
            expanded=True,
        )
        layout.addWidget(info_wrapper)
        
        layout.addStretch(1)
        scroll_area.setWidget(panel)
        
        return scroll_area

    @staticmethod
    def _resolve_panel_title(panel: QWidget) -> str:
        for attr in ("custom_title", "windowTitle", "title"):
            candidate = getattr(panel, attr, None)
            if callable(candidate):
                try:
                    value = candidate()
                except Exception:
                    value = None
            else:
                value = candidate
            if isinstance(value, str) and value.strip():
                return value.strip()
        return panel.__class__.__name__

    def add_custom_panel(self, panel: QWidget, index: int = 2, side: str = 'left'):
        """Add a custom panel to either the left or right sidebar."""
        if side == 'left' and hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)
        elif side == 'right' and hasattr(self, 'info_panel_layout'):
            wrapped_panel = panel
            if not isinstance(panel, CollapsiblePanel):
                title = self._resolve_panel_title(panel)
                wrapped_panel = CollapsiblePanel(title=title, content=panel, expanded=True)

            # Insert before the stretch (last item)
            count = self.info_panel_layout.count()
            # If there is a stretch, insert before it. 
            # The stretch is usually the last item.
            if count > 0:
                self.info_panel_layout.insertWidget(count - 1, wrapped_panel)
            else:
                self.info_panel_layout.addWidget(wrapped_panel)

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

    def set_data(self, data: VolumeData, reset_camera: bool = False, preserve_overlays: bool = False):
        """Set data and delegate to RenderEngine."""
        if not preserve_overlays:
            self._on_clear_overlays(silent=True)
        previous_mode = self.active_view_mode
        self.render_engine.set_data(data)
        
        # Sync with ROIHandler
        if hasattr(self, 'roi_handler'):
            self.roi_handler.set_data(data, self.render_engine.grid)
        
        d_type = data.metadata.get('Type', 'Unknown')

        if data.has_mesh:
            self.update_status(f"Loaded Mesh: {d_type}")
            self.render_mesh(reset_view=reset_camera)
            self.info_panel.update_info(d_type, (0, 0, 0), data.spacing, data.metadata)
        elif data.raw_data is not None:
            self.update_status(f"Loaded Volume: {d_type}")

            is_processed = ("Processed" in d_type)
            default_color = "red" if is_processed else "ivory"
            idx = self.params_panel.solid_color_combo.findText(default_color)
            if idx >= 0:
                with QSignalBlocker(self.params_panel.solid_color_combo):
                    self.params_panel.solid_color_combo.setCurrentIndex(idx)
            
            # Update Histogram
            self._update_histogram()

            min_val = np.nanmin(data.raw_data)
            max_val = np.nanmax(data.raw_data)
            self.params_panel.set_data_range(min_val, max_val)

            dims = self.render_engine.grid.dimensions
            self.params_panel.set_slice_limits(dims[0] - 1, dims[1] - 1, dims[2] - 1)
            self.params_panel.set_slice_defaults(dims[0] // 2, dims[1] // 2, dims[2] // 2)

            # Keep user's current volumetric view mode during time-step updates.
            target_mode = 'volume'
            if not reset_camera and previous_mode in {'volume', 'slices', 'iso'}:
                target_mode = previous_mode

            if target_mode == 'slices':
                self.render_slices(reset_view=False)
            elif target_mode == 'iso':
                self.render_isosurface_auto(reset_view=False)
            else:
                self.render_volume(reset_view=reset_camera)
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

    def set_timeseries_handler(self, handler) -> None:
        """Inject 4D handler so main window can resolve per-timepoint sources."""
        self._timeseries_handler = handler

    # ==========================================
    # Helper Methods for Data Operations
    # ==========================================
    
    def _clear_render_caches(self):
        """Clear all render engine caches via unified RenderEngine teardown."""
        if hasattr(self, 'render_engine'):
            self.render_engine.release_resources(
                clear_scene=True,
                clear_data=True,
                clear_cache=True,
                clear_gpu=True,
                add_axes=True,
            )
    
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
    # Overlay Helpers
    # ==========================================

    def _resolve_overlay_volume(self, source_name: str) -> Optional[VolumeData]:
        """Resolve UI source selection to a volume object."""
        if not self._data_manager:
            return None

        normalized = str(source_name or "").strip().lower()
        if normalized == "raw ct":
            ts = self._timeseries_handler
            if ts is not None and getattr(ts, "has_volumes", False):
                current = ts.get_current_raw_volume()
                if current is not None:
                    return current
            return self._data_manager.raw_ct_data
        if normalized == "segmented":
            ts = self._timeseries_handler
            if ts is not None and getattr(ts, "has_volumes", False):
                ts_panel = getattr(ts, "panel", None)
                threshold_getter = getattr(ts_panel, "get_threshold", None)
                threshold = int(threshold_getter() if callable(threshold_getter) else 0)
                return ts.get_or_request_segmented_overlay(
                    threshold=threshold,
                    on_ready=self._replay_overlays,
                )
            return self._data_manager.segmented_volume
        return None

    def _add_overlay_layer(
        self,
        layer_key: str,
        source_name: str,
        opacity: float,
        *,
        track_state: bool = True,
        silent: bool = False,
    ) -> bool:
        layer = str(layer_key or "").strip().lower()
        source = str(source_name or "").strip().lower()
        alpha = float(max(0.0, min(1.0, opacity)))

        # ROI is intentionally excluded from overlay sources.
        if source == "roi":
            if not silent:
                self.update_status("Overlay skipped: ROI source is disabled.")
            return False

        if layer == "pnm_mesh":
            pnm_data = self._data_manager.pnm_model if self._data_manager else None
            if pnm_data is None or not pnm_data.has_mesh:
                if not silent:
                    self.update_status("Overlay skipped: PNM mesh is not available.")
                return False
            success = self.render_engine.add_overlay_layer(layer, opacity=alpha, mesh_data=pnm_data)
        else:
            overlay_volume = self._resolve_overlay_volume(source_name)
            if overlay_volume is None or overlay_volume.raw_data is None:
                if not silent:
                    self.update_status(f"Overlay skipped: source '{source_name}' is not available.")
                return False
            if not self.render_engine.set_overlay_volume(overlay_volume):
                if not silent:
                    self.update_status("Overlay skipped: failed to prepare overlay volume.")
                return False
            success = self.render_engine.add_overlay_layer(layer, opacity=alpha)

        if not success:
            if not silent:
                self.update_status(f"Overlay failed: {layer_key}.")
            return False

        if track_state:
            descriptor = {"layer": layer, "source": source_name, "opacity": alpha}
            if descriptor not in self._active_overlay_layers:
                self._active_overlay_layers.append(descriptor)

        if not silent:
            self.update_status(f"Overlay added: {layer_key}")
        return True

    def _on_add_overlay_layer(self, layer_key: str, source_name: str, opacity: float):
        layer = str(layer_key or "").strip().lower()
        alpha = float(max(0.0, min(1.0, opacity)))
        descriptor = {"layer": layer, "source": source_name, "opacity": alpha}
        if descriptor not in self._active_overlay_layers:
            self._active_overlay_layers.append(descriptor)
        self._add_overlay_layer(layer_key, source_name, alpha, track_state=False, silent=False)

    def _on_clear_overlays(self, silent: bool = False):
        self._active_overlay_layers.clear()
        if hasattr(self, "render_engine"):
            self.render_engine.remove_overlay_layers()
        if not silent:
            self.update_status("Overlay layers cleared.")

    def _replay_overlays(self):
        if self._replaying_overlays or not self._active_overlay_layers:
            return

        self._replaying_overlays = True
        try:
            requested_layers = list(self._active_overlay_layers)
            if hasattr(self, "render_engine"):
                # Ensure idempotent replay even when base view used an in-place
                # update path (for example volume property tweaks).
                self.render_engine.remove_overlay_layers(render=False)

            for item in requested_layers:
                self._add_overlay_layer(
                    layer_key=str(item.get("layer", "")),
                    source_name=str(item.get("source", "Raw CT")),
                    opacity=float(item.get("opacity", 0.35)),
                    track_state=False,
                    silent=True,
                )
        finally:
            self._replaying_overlays = False

    # ==========================================
    # Rendering Delegation
    # ==========================================

    def clear_view(self):
        self._on_clear_overlays(silent=True)
        self.render_engine.clear_view()

    def reset_camera(self):
        self.render_engine.reset_camera()

    def render_mesh(self, reset_view=True):
        self.render_engine.render_mesh(reset_view=reset_view)
        self._replay_overlays()

    def render_volume(self, reset_view=True):
        self.render_engine.render_volume(reset_view=reset_view)
        self._replay_overlays()

    def render_slices(self, reset_view=True):
        self.render_engine.render_slices(reset_view=reset_view)
        self._replay_overlays()

    def render_isosurface_auto(self, reset_view=True):
        self.render_engine.render_isosurface_auto(reset_view=reset_view)
        self._replay_overlays()

    def render_isosurface(self, threshold=300, reset_view=True):
        self.render_engine.render_isosurface(threshold=threshold, reset_view=reset_view)
        self._replay_overlays()

    def _sync_render_style_ui(self):
        """Sync render-style combo to engine state without emitting UI signals."""
        if not hasattr(self, 'params_panel') or not hasattr(self, 'render_engine'):
            return
        combo = self.params_panel.render_style_combo
        desired = self.render_engine.get_current_render_mode_label()
        index = combo.findText(desired)
        if index >= 0 and combo.currentIndex() != index:
            with QSignalBlocker(combo):
                combo.setCurrentIndex(index)

    def _on_render_style_changed(self, render_style: str):
        """
        Handle render-style requests with central state + guard clauses.
        Falls back to full render only when no live actor is available.
        """
        result = self.render_engine.request_render_mode_change(render_style)
        if result == 'unchanged':
            return
        if self.active_view_mode == 'iso' and result != 'applied':
            self.render_isosurface_auto(reset_view=False)

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

        import traceback

        min_val, max_val = float(clim[0]), float(clim[1])
        ts = self._timeseries_handler
        if ts is not None and getattr(ts, "has_volumes", False):
            def on_transform_done(result: Dict[str, Any]) -> None:
                current_min = int(result.get("current_data_min", min_val))
                current_max = int(result.get("current_data_max", max_val))
                self.params_panel.set_data_range(current_min, current_max)
                self._update_histogram()
                self._refresh_view()
                if result.get("cancelled", False):
                    self.update_status("4D clip cancelled (applied frames kept).")
                else:
                    self.update_status(f"Applied 4D clip: [{min_val:.0f}, {max_val:.0f}]")

            ts.run_series_transform(
                mode="clip",
                min_val=min_val,
                max_val=max_val,
                completion_callback=on_transform_done,
            )
            return

        try:
            # Step 1: Clear render caches
            self.update_status("Clipping: Clearing caches...")
            self._clear_render_caches()

            # Step 2: Delegate data manipulation to DataManager.
            # Use a weakref so the callback does not keep the window alive if
            # the UI is closed while a long clip operation is running.
            _self_ref = weakref.ref(self)

            def progress_cb(percent, msg):
                win = _self_ref()
                if win is not None:
                    win.update_status(f"Clipping: {msg}")

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

        except ValueError as e:
            self.update_status(str(e))
        except MemoryError as e:
            print(f"[RangeClip] Memory Error: {e}")
            print(traceback.format_exc())
            self.update_status("Memory Error during clip")
        except Exception as e:
            print(f"[RangeClip] Error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            self.update_status(f"Clip error: {type(e).__name__}")

    def _on_invert_volume(self):
        """Invert volume values (delegates to DataManager)."""
        if not hasattr(self, '_data_manager') or not self._data_manager:
            self.update_status("No data to invert")
            return

        import traceback
        current_clim = self.params_panel.get_current_values().get('clim', [0, 1000])
        old_min_clim, old_max_clim = float(current_clim[0]), float(current_clim[1])
        ts = self._timeseries_handler
        if ts is not None and getattr(ts, "has_volumes", False):
            def on_transform_done(result: Dict[str, Any]) -> None:
                data_min = float(result.get("current_data_min", old_min_clim))
                data_max = float(result.get("current_data_max", old_max_clim))
                invert_offset = float(result.get("invert_offset", data_min + data_max))
                new_min_clim = int(invert_offset - old_max_clim)
                new_max_clim = int(invert_offset - old_min_clim)

                self.params_panel.set_data_range(int(data_min), int(data_max))
                self.params_panel.block_signals(True)
                self.params_panel.slider_clim_min.setValue(new_min_clim)
                self.params_panel.slider_clim_max.setValue(new_max_clim)
                self.params_panel.spinbox_clim_min.setValue(new_min_clim)
                self.params_panel.spinbox_clim_max.setValue(new_max_clim)
                self.params_panel.block_signals(False)
                self._update_histogram()
                self._refresh_view()

                if result.get("cancelled", False):
                    self.update_status("4D invert cancelled (applied frames kept).")
                else:
                    self.update_status(f"4D volume inverted. Clim: [{new_min_clim}, {new_max_clim}]")

            ts.run_series_transform(
                mode="invert",
                completion_callback=on_transform_done,
            )
            return

        try:
            # Step 1: Clear render caches
            self.update_status("Inverting: Clearing caches...")
            self._clear_render_caches()

            # Step 2: Get current clim before inversion (for UI update)
            data = self._data_manager.active_data
            old_min_clim, old_max_clim = current_clim[0], current_clim[1]

            # Step 3: Delegate data inversion to DataManager.
            # Use weakref to avoid keeping the window alive through the closure.
            _self_ref = weakref.ref(self)

            def progress_cb(percent, msg):
                win = _self_ref()
                if win is not None:
                    win.update_status(f"Inverting: {msg}")

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

    def closeEvent(self, event: QCloseEvent):
        """Explicitly release rendering resources when window closes."""
        try:
            if hasattr(self, "render_engine"):
                self.render_engine.release_resources(
                    clear_scene=True,
                    clear_data=True,
                    clear_cache=True,
                    clear_gpu=True,
                    add_axes=False,
                )
        except Exception:
            pass

        try:
            if hasattr(self, "plotter"):
                self.plotter.close()
        except Exception:
            pass

        super().closeEvent(event)

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
            raw_z, raw_y, raw_x = world_xyz_to_index_zyx(
                (float(world_pos[0]), float(world_pos[1]), float(world_pos[2])),
                self.data.spacing,
                self.data.origin,
                rounding="round",
            )

            shape = self.data.raw_data.shape
            if 0 <= raw_z < shape[0] and 0 <= raw_y < shape[1] and 0 <= raw_x < shape[2]:
                val = self.data.raw_data[raw_z, raw_y, raw_x]
                self.status_bar.showMessage(
                    f"馃搷 Pos: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}) | "
                    f"Indices: [{raw_z}, {raw_y}, {raw_x}] | 馃挕 Intensity: {val:.1f}"
                )
        except Exception:
            pass

