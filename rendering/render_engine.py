"""
Core rendering engine for volumetric data visualization.
Provides reusable rendering methods independent of GUI framework.
Includes LOD (Level of Detail) support for large volume handling.
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pyvista as pv
from core import VolumeData
from core.dto import RenderParamsDTO
from core.coordinates import raw_zyx_to_grid_xyz
from rendering.lod_manager import LODPyramid, LODRenderManager, check_gpu_volume_rendering
from config import (
    RENDER_MAX_VOXELS_VOLUME,
    RENDER_MAX_VOXELS_ISO,
    RENDER_MAX_MEMORY_MB
)

# Aliases for backward compatibility
MAX_VOXELS_FOR_VOLUME = RENDER_MAX_VOXELS_VOLUME
MAX_VOXELS_FOR_ISO = RENDER_MAX_VOXELS_ISO
MAX_MEMORY_MB_RENDER = RENDER_MAX_MEMORY_MB


class RenderEngine:
    """
    Handles PyVista rendering logic.
    Designed for composition with GUI classes.
    """

    def __init__(self, plotter, params_panel, info_panel=None, clip_panel=None, status_callback=None):
        """
        Initialize engine with renderer dependencies.
        
        Args:
            plotter: BackgroundPlotter instance
            params_panel: RenderingParametersPanel for current parameters
            info_panel: Optional InfoPanel for data display
            clip_panel: Optional ClipPlanePanel for clipping controls
            status_callback: Optional callback for status updates
        """
        self.plotter = plotter
        self.params_panel = params_panel
        self.info_panel = info_panel
        self.clip_panel = clip_panel
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
        self._lod_pyramid: Optional[LODPyramid] = None
        self._overlay_grid: Optional[pv.ImageData] = None
        self._overlay_source_token: Optional[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]] = None
        self._overlay_actors: List[Any] = []
        
        # PNM color range cache (for consistent colors across timepoints)
        self._pnm_radius_clim: Optional[Tuple[float, float]] = None
        self._pnm_compression_clim: Optional[Tuple[float, float]] = None
        self._highlight_pore_id: Optional[int] = None
        
        # GPU capability
        self.gpu_available, self.gpu_info = check_gpu_volume_rendering()

        # Actors
        self.volume_actor = None
        self.iso_actor = None
        self._current_iso_threshold: Optional[int] = None

        # Centralized render-style state (single source of truth for isosurface style)
        self._current_render_mode = 'surface'
        self._current_render_show_edges = False
        self._current_render_mode_label = 'Surface'

        # Injected params DTO (used when running headless without a params_panel)
        self._injected_params: Optional[RenderParamsDTO] = None

    # ------------------------------------------------------------------
    # Headless / DTO interface
    # ------------------------------------------------------------------

    def inject_params(self, dto: RenderParamsDTO) -> None:
        """
        Push a RenderParamsDTO directly into the engine.

        When set, ``_get_render_params()`` will return values from this DTO
        instead of querying the live params_panel.  Call with ``None`` to
        restore live-panel mode.
        """
        self._injected_params = dto

    def _get_render_params(self) -> Dict[str, Any]:
        """
        Return current render parameters as a plain dict.

        Priority:
        1. Injected DTO (headless / CLI mode)
        2. Live params_panel.get_current_values() (GUI mode)
        3. Hard-coded safe defaults (neither present)
        """
        if self._injected_params is not None:
            return self._injected_params.to_dict()
        if self.params_panel is not None:
            return self.params_panel.get_current_values()
        return RenderParamsDTO().to_dict()

    # ------------------------------------------------------------------

    def update_status(self, message: str):
        """Update status via callback."""
        if self._status_callback:
            self._status_callback(message)
        else:
            print(f"[RenderEngine] {message}")

    @staticmethod
    def _detach_mapper_inputs(mapper) -> None:
        """Best-effort disconnect of VTK mapper input graph."""
        if mapper is None:
            return

        for method_name, args in (
            ("SetInputConnection", (None,)),
            ("SetInputDataObject", (None,)),
            ("SetInputData", (None,)),
            ("RemoveAllInputs", ()),
            ("RemoveAllClippingPlanes", ()),
        ):
            method = getattr(mapper, method_name, None)
            if callable(method):
                try:
                    method(*args)
                except Exception:
                    pass

    def _detach_actor(self, actor) -> None:
        """Best-effort disconnect of actor -> mapper -> data references."""
        if actor is None:
            return

        mapper = getattr(actor, "mapper", None)
        if mapper is None:
            get_mapper = getattr(actor, "GetMapper", None)
            if callable(get_mapper):
                try:
                    mapper = get_mapper()
                except Exception:
                    mapper = None

        self._detach_mapper_inputs(mapper)

        set_mapper = getattr(actor, "SetMapper", None)
        if callable(set_mapper):
            try:
                set_mapper(None)
            except Exception:
                pass

        if hasattr(actor, "mapper"):
            try:
                actor.mapper = None
            except Exception:
                pass

    def _prepare_vtk_volume_array(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Normalize volume layout before handing memory to VTK.

        We explicitly repack non-contiguous / reversed-stride views so memory
        cost is predictable and not deferred to implicit bridge conversions.
        """
        array = np.asarray(raw_data)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape={array.shape}")

        has_nonpositive_stride = any(stride <= 0 for stride in array.strides)
        is_contiguous = array.flags.c_contiguous or array.flags.f_contiguous
        if array.flags.f_contiguous and not has_nonpositive_stride:
            return array

        est_mb = array.nbytes / (1024 * 1024)
        if not is_contiguous or has_nonpositive_stride:
            self.update_status(
                f"Repacking non-contiguous volume for VTK (~{est_mb:.1f} MB copy)."
            )
        else:
            self.update_status(
                f"Converting volume to Fortran layout for VTK (~{est_mb:.1f} MB copy)."
            )
        return np.asfortranarray(array)

    def release_resources(
        self,
        *,
        clear_scene: bool = True,
        clear_data: bool = True,
        clear_cache: bool = True,
        clear_gpu: bool = True,
        add_axes: bool = True,
    ) -> None:
        """
        Explicitly tear down VTK/PyVista resources in reverse dependency order.
        """
        self.remove_overlay_layers(render=False)

        # 1) Detach actors from renderer and disconnect mapper inputs first.
        for actor_attr in ("volume_actor", "iso_actor"):
            actor = getattr(self, actor_attr, None)
            if actor is None:
                continue
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
            self._detach_actor(actor)
            setattr(self, actor_attr, None)

        self._current_iso_threshold = None

        # 2) Optionally clear scene-level objects.
        if clear_scene:
            try:
                self.plotter.clear()
            except Exception:
                pass
            if add_axes:
                try:
                    self.plotter.add_axes()
                except Exception:
                    pass
            self.active_view_mode = None
            if self.params_panel:
                try:
                    self.params_panel.set_mode(None)
                except Exception:
                    pass

        # 3) Drop intermediate caches and LOD references.
        if clear_cache:
            self._iso_cache = {}
            self._cached_vol_grid = None
            self._cached_vol_grid_source = None
            self._lod_pyramid = None
            self._overlay_grid = None
            self._overlay_source_token = None

        # 4) Release core data references after graph disconnect.
        if clear_data:
            self.data = None
            self.grid = None
            self.mesh = None
            self.active_view_mode = None
            self._pnm_radius_clim = None
            self._pnm_compression_clim = None
            self._highlight_pore_id = None

            if clear_gpu:
                try:
                    from core.gpu_backend import get_gpu_backend
                    get_gpu_backend().clear_memory(force=True)
                except Exception:
                    pass

    def teardown(self) -> None:
        """Compatibility alias for full engine teardown."""
        self.release_resources(
            clear_scene=True,
            clear_data=True,
            clear_cache=True,
            clear_gpu=True,
            add_axes=False,
        )

    def set_data(self, data: VolumeData):
        """Set volume data and prepare grid. Clears previous data first."""
        # Explicitly release previous references — no gc.collect() needed;
        # Python's reference counting handles deallocation as soon as the
        # last reference drops.
        had_previous = any((
            self.data is not None,
            self.grid is not None,
            self.mesh is not None,
            self.volume_actor is not None,
            self.iso_actor is not None,
        ))
        if had_previous:
            self.update_status("Clearing previous data...")

        self.release_resources(
            clear_scene=had_previous,
            clear_data=True,
            clear_cache=True,
            clear_gpu=had_previous,
            add_axes=had_previous,
        )

        # Now set new data
        self.data = data

        if data.has_mesh:
            self.mesh = data.mesh
        elif data.raw_data is not None:
            self._create_pyvista_grid()
            # Create LOD pyramid for large volumes
            if self.grid and self.grid.n_cells > MAX_VOXELS_FOR_VOLUME:
                self._lod_pyramid = LODPyramid(self.grid, levels=3)
                self.update_status(f"Created LOD pyramid: {self._lod_pyramid}")

    def _create_pyvista_grid(self):
        """Create PyVista ImageData grid from volume data."""
        if not self.data or self.data.raw_data is None:
            return
        prepared = self._prepare_vtk_volume_array(self.data.raw_data)
        grid = self._build_grid_from_volume_data(prepared, self.data.spacing, self.data.origin)
        self.grid = grid

    def _build_grid_from_volume_data(
        self,
        raw_data: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> pv.ImageData:
        """
        Create a PyVista ImageData object from raw volume data.

        Input raw_data is expected in project storage order (z, y, x) and is
        converted to VTK axis order (x, y, z).
        """
        raw_xyz = raw_zyx_to_grid_xyz(raw_data)
        if not raw_xyz.flags.f_contiguous:
            raw_xyz = np.asfortranarray(raw_xyz)

        grid = pv.ImageData()
        grid.dimensions = np.array(raw_xyz.shape) + 1
        grid.origin = origin
        grid.spacing = spacing
        cell_values = raw_xyz.ravel(order="F")
        if not cell_values.flags.c_contiguous:
            cell_values = np.ascontiguousarray(cell_values)
        grid.cell_data["values"] = cell_values
        return grid

    def set_overlay_volume(self, volume_data: Optional[VolumeData]) -> bool:
        """
        Convert selected overlay source volume into cached ImageData.

        Returns True when an overlay grid is available and ready.
        """
        if volume_data is None or volume_data.raw_data is None:
            self._overlay_grid = None
            self._overlay_source_token = None
            return False

        source_token = (
            id(volume_data.raw_data),
            tuple(float(v) for v in volume_data.spacing),
            tuple(float(v) for v in volume_data.origin),
        )
        if self._overlay_grid is not None and self._overlay_source_token == source_token:
            return True

        prepared = self._prepare_vtk_volume_array(volume_data.raw_data)
        self._overlay_grid = self._build_grid_from_volume_data(
            prepared,
            spacing=volume_data.spacing,
            origin=volume_data.origin,
        )
        self._overlay_source_token = source_token
        return True

    def add_overlay_layer(
        self,
        layer_type: str,
        *,
        opacity: float = 0.35,
        mesh_data: Optional[VolumeData] = None,
    ) -> bool:
        """
        Add one overlay actor on top of the active base view without clearing scene.
        """
        overlay_type = str(layer_type or "").strip().lower()
        alpha = float(max(0.0, min(1.0, opacity)))
        params = self._get_render_params()
        actor = None

        if overlay_type == "pnm_mesh":
            mesh_obj = None
            if mesh_data is not None and mesh_data.has_mesh:
                mesh_obj = mesh_data.mesh
            elif self.mesh is not None:
                mesh_obj = self.mesh
            if mesh_obj is None:
                return False

            actor = self.plotter.add_mesh(
                mesh_obj,
                color=params.get("solid_color", "gold"),
                opacity=alpha,
                smooth_shading=True,
                show_scalar_bar=False,
                render=False,
            )
        else:
            if self._overlay_grid is None:
                return False

            if overlay_type == "slices":
                ox, oy, oz = self._overlay_grid.origin
                dx, dy, dz = self._overlay_grid.spacing
                x = ox + params.get("slice_x", 0) * dx
                y = oy + params.get("slice_y", 0) * dy
                z = oz + params.get("slice_z", 0) * dz
                slices = self._overlay_grid.slice_orthogonal(x=x, y=y, z=z)
                actor = self.plotter.add_mesh(
                    slices,
                    cmap=params.get("colormap", "bone"),
                    clim=params.get("clim", [0, 1000]),
                    opacity=alpha,
                    show_scalar_bar=False,
                    render=False,
                )
            elif overlay_type == "iso":
                threshold = params.get("threshold", 300)
                contours = self._overlay_grid.cell_data_to_point_data().contour(isosurfaces=[threshold])
                if contours.n_points == 0:
                    return False
                iso_style = self.get_iso_mesh_kwargs(params=params, sync_mode=False)
                actor = self.plotter.add_mesh(
                    contours,
                    color=params.get("solid_color", "ivory"),
                    opacity=alpha,
                    show_scalar_bar=False,
                    render=False,
                    **iso_style,
                )
            elif overlay_type == "volume":
                vol_grid = self._overlay_grid.cell_data_to_point_data()
                opacity_tf = np.linspace(0.0, alpha, 8).tolist()
                actor = self.plotter.add_volume(
                    vol_grid,
                    cmap=params.get("colormap", "bone"),
                    clim=params.get("clim", [0, 1000]),
                    opacity=opacity_tf,
                    shade=False,
                    render=False,
                )
            else:
                return False

        if actor is None:
            return False

        self._overlay_actors.append(actor)
        self.plotter.render()
        return True

    def remove_overlay_layers(self, *, render: bool = True) -> None:
        """Remove all overlay actors from the scene and detach references."""
        if not self._overlay_actors:
            return

        for actor in self._overlay_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
            self._detach_actor(actor)
        self._overlay_actors.clear()

        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def clear_view(self):
        """Clear rendered actors but keep loaded data and caches."""
        self.release_resources(
            clear_scene=True,
            clear_data=False,
            clear_cache=False,
            clear_gpu=False,
            add_axes=True,
        )

    def set_highlight_pore(self, pore_id: Optional[int]):
        """Set the pore ID to highlight in yellow. Pass None to clear highlight."""
        self._highlight_pore_id = pore_id
    
    def clear_pnm_color_cache(self):
        """Clear PNM color range cache (call when loading new dataset)."""
        self._pnm_radius_clim = None
        self._pnm_compression_clim = None
        self._highlight_pore_id = None

    def reset_camera(self):
        """Reset camera to isometric view."""
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def update_clim_fast(self, clim: Tuple[float, float]) -> bool:
        """Fast update of scalar range without re-rendering volume.
        
        Args:
            clim: Tuple of (min, max) scalar values.
            
        Returns:
            True if update was successful, False if full re-render is needed.
        """
        if self.volume_actor is None:
            return False
        
        try:
            # Try to update mapper scalar range directly
            if hasattr(self.volume_actor, 'mapper') and self.volume_actor.mapper:
                self.volume_actor.mapper.scalar_range = clim
                self.plotter.render()
                return True
        except Exception as e:
            print(f"[RenderEngine] Fast clim update failed: {e}")
        
        return False

    def _get_downsampled_grid(self, max_voxels: int) -> Optional[pv.ImageData]:
        """Get grid with automatic downsampling if exceeds voxel threshold.
        
        Args:
            max_voxels: Maximum allowed voxels before downsampling.
            
        Returns:
            Original or downsampled grid.
        """
        if not self.grid:
            return None
        
        n_cells = self.grid.n_cells
        if n_cells <= max_voxels:
            return self.grid
        
        # Calculate downsampling step (cubic root to reduce in all dimensions)
        step = int(np.ceil((n_cells / max_voxels) ** (1/3)))
        step = max(2, step)  # Minimum step of 2
        dims = self.grid.dimensions
        
        try:
            # Extract every Nth cell in each dimension using keyword args
            downsampled = self.grid.extract_subset(
                voi=(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1),
                rate=(step, step, step)
            )
            self.update_status(f"Auto-downsampled for memory: {n_cells:,} -> {downsampled.n_cells:,} cells (step={step})")
            return downsampled
        except Exception as e:
            print(f"[RenderEngine] Downsampling failed: {e}")
            return self.grid

    def _get_grid_for_isosurface(self) -> Optional[pv.ImageData]:
        """Get grid suitable for isosurface rendering."""
        # Use LOD pyramid if available
        if self._lod_pyramid and self._lod_pyramid.num_levels > 1:
            return self._lod_pyramid.get_for_memory(MAX_MEMORY_MB_RENDER)
        return self._get_downsampled_grid(MAX_VOXELS_FOR_ISO)
    
    def _get_grid_for_volume(self) -> Optional[pv.ImageData]:
        """Get grid suitable for volume rendering using LOD if available."""
        # Use LOD pyramid if available (pre-computed levels)
        if self._lod_pyramid and self._lod_pyramid.num_levels > 1:
            grid = self._lod_pyramid.get_for_memory(MAX_MEMORY_MB_RENDER)
            if grid:
                self.update_status(f"Using LOD level: {grid.n_cells:,} cells")
                return grid
        return self._get_downsampled_grid(MAX_VOXELS_FOR_VOLUME)

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

    @staticmethod
    def normalize_render_mode(render_style: Optional[str]) -> Tuple[str, bool, str]:
        """Normalize UI render style text into VTK style + edge-visibility flags."""
        style_map = {
            'surface': ('surface', False, 'Surface'),
            'wireframe': ('wireframe', False, 'Wireframe'),
            'wireframe + surface': ('surface', True, 'Wireframe + Surface'),
            'points': ('points', False, 'Points'),
        }
        key = str(render_style or 'Surface').strip().lower()
        return style_map.get(key, ('surface', False, 'Surface'))

    def get_current_render_mode_label(self) -> str:
        """Return current UI label for isosurface render style."""
        return self._current_render_mode_label

    def get_iso_mesh_kwargs(self, params: Optional[Dict[str, Any]] = None, sync_mode: bool = True) -> Dict[str, Any]:
        """
        Build shared mesh kwargs for isosurface-like rendering.

        This centralizes style mapping to keep render paths DRY.
        """
        render_style = self._current_render_mode_label
        if isinstance(params, dict):
            render_style = params.get('render_style', render_style)

        style, show_edges, canonical = self.normalize_render_mode(render_style)
        if sync_mode:
            self._current_render_mode = style
            self._current_render_show_edges = show_edges
            self._current_render_mode_label = canonical

        return {
            'style': style,
            'show_edges': show_edges,
            'smooth_shading': True,
            'specular': 0.4,
            'diffuse': 0.7,
            'ambient': 0.15,
            'lighting': True
        }

    def _apply_actor_render_style(self, actor, style: str, show_edges: bool) -> bool:
        """Apply style in-place to an existing actor without rebuilding scene objects."""
        prop = getattr(actor, 'prop', None)
        if prop is None:
            try:
                prop = actor.GetProperty()
            except Exception:
                prop = None
        if prop is None:
            return False

        changed = False
        try:
            # PyVista actor property helper
            prop.style = style
            changed = True
        except Exception:
            try:
                # VTK fallback
                if style == 'wireframe':
                    prop.SetRepresentationToWireframe()
                elif style == 'points':
                    prop.SetRepresentationToPoints()
                else:
                    prop.SetRepresentationToSurface()
                changed = True
            except Exception:
                pass

        try:
            if hasattr(prop, 'show_edges'):
                prop.show_edges = bool(show_edges)
                changed = True
            elif hasattr(prop, 'SetEdgeVisibility'):
                prop.SetEdgeVisibility(1 if show_edges else 0)
                changed = True
        except Exception:
            pass

        return changed

    def request_render_mode_change(self, mode_label: str) -> str:
        """
        Request an isosurface render-style change.

        Returns:
            'unchanged': requested mode equals current state
            'applied': changed and applied in-place to current iso actor
            'state_only': state changed but no live iso actor to mutate
        """
        style, show_edges, canonical = self.normalize_render_mode(mode_label)
        if style == self._current_render_mode and show_edges == self._current_render_show_edges:
            return 'unchanged'

        self._current_render_mode = style
        self._current_render_show_edges = show_edges
        self._current_render_mode_label = canonical

        if self.active_view_mode == 'iso' and self.iso_actor is not None:
            if self._apply_actor_render_style(self.iso_actor, style, show_edges):
                self.plotter.render()
                return 'applied'
        return 'state_only'

    def render_mesh(self, reset_view=True):
        """Render PNM mesh with PoreRadius colormap."""
        if not self.mesh:
            return
        
        # Save camera state before clearing (for smooth time step transitions)
        saved_camera_position = None
        saved_camera_focal_point = None
        saved_camera_view_up = None
        saved_camera_view_angle = None
        
        if not reset_view and self.active_view_mode == 'mesh':
            try:
                # Save detailed camera state
                saved_camera_position = self.plotter.camera.position
                saved_camera_focal_point = self.plotter.camera.focal_point
                saved_camera_view_up = self.plotter.camera.up
                saved_camera_view_angle = self.plotter.camera.view_angle
            except Exception:
                pass
            
        # Clean up previous scene actors before adding new ones.
        self.clear_view()
        
        self.active_view_mode = 'mesh'
        self._update_clip_panel_state()
        if self.params_panel:
            self.params_panel.set_mode('mesh')
        self.plotter.enable_lightkit()

        params = self._get_render_params()

        # Priority 1: CompressionRatio (cached range) with optional highlight
        if "CompressionRatio" in self.mesh.point_data:
            comp = self.mesh.point_data["CompressionRatio"]
            if self._pnm_compression_clim is None:
                self._pnm_compression_clim = (float(comp.min()), float(comp.max()))
            clim_comp = self._pnm_compression_clim

            # Invert so smaller ratios look darker
            comp_display = 1.0 - comp
            clim_display = (1.0 - clim_comp[1], 1.0 - clim_comp[0])

            if self._highlight_pore_id is not None and "ID" in self.mesh.point_data:
                pore_ids = self.mesh.point_data.get("ID", np.array([]))
                highlight_mask = (pore_ids == self._highlight_pore_id)
                if highlight_mask.any():
                    display_scalar = comp_display.copy()
                    # set highlighted to max+10% to map to yellow
                    display_scalar[highlight_mask] = clim_display[1] * 1.1
                    self.plotter.add_mesh(
                        self.mesh,
                        scalars=display_scalar,
                        cmap=['black', 'dimgray', 'lightgray', 'yellow'],
                        clim=[clim_display[0], clim_display[1] * 1.1],
                        show_scalar_bar=True,
                        scalar_bar_args={'title': 'Compression (dark=compressed, yellow=selected)'},
                        smooth_shading=True,
                        specular=0.5
                    )
                    return

            # No highlight
            self.plotter.add_mesh(
                self.mesh,
                scalars=comp_display,
                cmap='Greys',
                clim=clim_display,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Compression (dark=compressed)'},
                smooth_shading=True,
                specular=0.5
            )

        # Priority 2: PoreRadius (cached range) with optional highlight
        elif "PoreRadius" in self.mesh.point_data:
            pore_radii = self.mesh.point_data["PoreRadius"]
            if self._pnm_radius_clim is None:
                self._pnm_radius_clim = (float(pore_radii.min()), float(pore_radii.max()))

            if self._highlight_pore_id is not None and "ID" in self.mesh.point_data:
                pore_ids = self.mesh.point_data.get("ID", np.array([]))
                highlight_mask = (pore_ids == self._highlight_pore_id)
                if highlight_mask.any():
                    display_scalar = pore_radii.copy()
                    max_radius = self._pnm_radius_clim[1]
                    display_scalar[highlight_mask] = max_radius * 1.5
                    self.plotter.add_mesh(
                        self.mesh,
                        scalars=display_scalar,
                        cmap=['darkblue', 'blue', 'cyan', 'lime', 'yellow'],
                        clim=[self._pnm_radius_clim[0], max_radius * 1.5],
                        show_scalar_bar=True,
                        scalar_bar_args={'title': 'Pore Radius (mm) - Yellow=Selected'},
                        smooth_shading=True,
                        specular=0.5
                    )
                    return

            # Normal rendering with cached color range
            self.plotter.add_mesh(
                self.mesh,
                scalars="PoreRadius",
                cmap=params.get('colormap', 'Oranges'),
                clim=self._pnm_radius_clim,
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
        elif saved_camera_position is not None:
            # Restore previous camera state with detailed parameters
            try:
                self.plotter.camera.position = saved_camera_position
                self.plotter.camera.focal_point = saved_camera_focal_point
                self.plotter.camera.up = saved_camera_view_up
                self.plotter.camera.view_angle = saved_camera_view_angle
                self.plotter.render()
            except Exception:
                pass

    def render_volume(self, reset_view=True):
        """
        Optimized Volume Rendering.
        Updates Opacity/Color in-place when possible.
        """
        if not self.grid:
            return

        # Save camera state before potential view change
        saved_camera_position = None
        saved_camera_focal_point = None
        saved_camera_view_up = None
        saved_camera_view_angle = None
        
        if not reset_view and self.active_view_mode in ['volume', 'mesh', 'slices', 'iso']:
            try:
                # Save detailed camera state
                saved_camera_position = self.plotter.camera.position
                saved_camera_focal_point = self.plotter.camera.focal_point
                saved_camera_view_up = self.plotter.camera.up
                saved_camera_view_angle = self.plotter.camera.view_angle
            except Exception:
                pass

        if reset_view or self.active_view_mode != 'volume' or self.volume_actor is None:
            self.update_status("Rendering volume (New)...")
            self.clear_view()
            self.active_view_mode = 'volume'
            self._update_clip_panel_state()
            if self.params_panel:
                self.params_panel.set_mode('volume')
            self.plotter.enable_lightkit()

            # Get potentially downsampled grid for memory safety
            safe_grid = self._get_grid_for_volume()
            if safe_grid is None:
                self.update_status("No grid available for volume rendering")
                return
            
            # Cache the point data grid
            grid_id = id(safe_grid)
            if self._cached_vol_grid is None or self._cached_vol_grid_source != grid_id:
                self._cached_vol_grid = safe_grid.cell_data_to_point_data()
                self._cached_vol_grid_source = grid_id

            vol_grid = self._cached_vol_grid
            params = self._get_render_params()

            self.volume_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params.get('colormap', 'bone'),
                opacity=params.get('opacity', 'sigmoid'),
                clim=params.get('clim', [0, 1000]),
                shade=False
            )
            self.plotter.add_axes()
            if reset_view:
                self.reset_camera()
            elif saved_camera_position is not None:
                # Restore previous camera state when switching from other modes
                try:
                    self.plotter.camera.position = saved_camera_position
                    self.plotter.camera.focal_point = saved_camera_focal_point
                    self.plotter.camera.up = saved_camera_view_up
                    self.plotter.camera.view_angle = saved_camera_view_angle
                    self.plotter.render()
                except Exception:
                    pass
        else:
            # Fast update without clearing
            self.update_status("Updating volume properties...")
            params   = self._get_render_params()
            vol_grid = self._cached_vol_grid or self.grid.cell_data_to_point_data()

            old_actor = self.volume_actor
            new_actor = self.plotter.add_volume(
                vol_grid,
                cmap=params.get('colormap', 'bone'),
                opacity=params.get('opacity', 'sigmoid'),
                clim=params.get('clim', [0, 1000]),
                shade=False,
                render=False
            )

            if new_actor.mapper:
                new_actor.mapper.scalar_range = params.get('clim', [0, 1000])

            if old_actor:
                self.plotter.remove_actor(old_actor, render=False)

            self.volume_actor = new_actor
            self.plotter.render()

    def render_slices(self, reset_view=True):
        """Render orthogonal slices."""
        if not self.grid:
            return

        # Save camera state before potential view change
        saved_camera_position = None
        saved_camera_focal_point = None
        saved_camera_view_up = None
        saved_camera_view_angle = None
        
        if not reset_view and self.active_view_mode in ['volume', 'mesh', 'slices', 'iso']:
            try:
                # Save detailed camera state
                saved_camera_position = self.plotter.camera.position
                saved_camera_focal_point = self.plotter.camera.focal_point
                saved_camera_view_up = self.plotter.camera.up
                saved_camera_view_angle = self.plotter.camera.view_angle
            except Exception:
                pass

        if reset_view or self.active_view_mode != 'slices':
            self.clear_view()
            self.active_view_mode = 'slices'
            self._update_clip_panel_state()
            if self.params_panel:
                self.params_panel.set_mode('slices')
            self.plotter.enable_lightkit()
            self.plotter.show_grid()

        params = self._get_render_params()
        ox, oy, oz = self.grid.origin
        dx, dy, dz = self.grid.spacing
        x = ox + params.get('slice_x', 0) * dx
        y = oy + params.get('slice_y', 0) * dy
        z = oz + params.get('slice_z', 0) * dz

        self.plotter.clear_actors()
        slices = self.grid.slice_orthogonal(x=x, y=y, z=z)
        self.plotter.add_mesh(
            slices,
            cmap=params.get('colormap', 'bone'),
            clim=params.get('clim', [0, 1000]),
            show_scalar_bar=False
        )
        self.plotter.add_axes()
        if reset_view:
            self.reset_camera()
        elif saved_camera_position is not None:
            # Restore previous camera state when switching from other modes
            try:
                self.plotter.camera.position = saved_camera_position
                self.plotter.camera.focal_point = saved_camera_focal_point
                self.plotter.camera.up = saved_camera_view_up
                self.plotter.camera.view_angle = saved_camera_view_angle
                self.plotter.render()
            except Exception:
                pass

    def render_isosurface(self, threshold=300, reset_view=True):
        """Render isosurface at specified threshold using optimized VTK algorithm."""
        if not self.grid:
            return

        # Save camera state before potential view change
        saved_camera_position = None
        saved_camera_focal_point = None
        saved_camera_view_up = None
        saved_camera_view_angle = None
        
        if not reset_view and self.active_view_mode in ['volume', 'mesh', 'slices', 'iso']:
            try:
                # Save detailed camera state
                saved_camera_position = self.plotter.camera.position
                saved_camera_focal_point = self.plotter.camera.focal_point
                saved_camera_view_up = self.plotter.camera.up
                saved_camera_view_angle = self.plotter.camera.view_angle
            except Exception:
                pass

        self.update_status(f"Generating isosurface ({threshold})...")
        self.clear_view()
        self.active_view_mode = 'iso'
        self._update_clip_panel_state()
        if self.params_panel:
            self.params_panel.set_mode('iso')
        self.plotter.enable_lightkit()
        params = self._get_render_params()
        mesh_kwargs = self.get_iso_mesh_kwargs(params=params, sync_mode=True)

        try:
            # Check cache
            if threshold in self._iso_cache:
                contours = self._iso_cache[threshold]
                self.update_status("Using cached isosurface")
            else:
                import time
                start = time.time()
                
                # Use potentially downsampled grid for large volumes
                safe_grid = self._get_grid_for_isosurface()
                if safe_grid is None:
                    self.update_status("No grid available for isosurface")
                    return
                
                grid_points = safe_grid.cell_data_to_point_data()
                
                # Use VTK Flying Edges algorithm (faster than marching cubes)
                # flying_edges is 2-10x faster and is the default in VTK 9.0+
                try:
                    # Flying Edges via VTK directly
                    import vtk
                    contour_filter = vtk.vtkFlyingEdges3D()
                    contour_filter.SetInputData(grid_points)
                    contour_filter.SetValue(0, threshold)
                    contour_filter.ComputeNormalsOn()
                    contour_filter.Update()
                    contours = pv.wrap(contour_filter.GetOutput())
                    elapsed = time.time() - start
                    print(f"[Render] Isosurface (FlyingEdges): {elapsed:.2f}s, {contours.n_points} points")
                except Exception as e:
                    # Fallback to PyVista contour
                    print(f"[Render] FlyingEdges failed ({e}), using PyVista contour")
                    contours = grid_points.contour(isosurfaces=[threshold])
                    contours.compute_normals(inplace=True)
                    elapsed = time.time() - start
                    print(f"[Render] Isosurface (contour): {elapsed:.2f}s, {contours.n_points} points")
                
                self._iso_cache[threshold] = contours

            mode = params.get('coloring_mode', 'Solid Color')
            actor = None
            if mode == 'Solid Color':
                actor = self.plotter.add_mesh(contours, color=params.get('solid_color', 'ivory'), **mesh_kwargs)
            elif mode == 'Depth (Z-Axis)':
                contours["Elevation"] = contours.points[:, 2]
                actor = self.plotter.add_mesh(contours, scalars="Elevation", cmap=params.get('colormap', 'viridis'), **mesh_kwargs)
            elif mode == 'Radial (Center Dist)':
                dist = np.linalg.norm(contours.points - contours.center, axis=1)
                contours["RadialDistance"] = dist
                actor = self.plotter.add_mesh(contours, scalars="RadialDistance", cmap=params.get('colormap', 'viridis'), **mesh_kwargs)

            self.iso_actor = actor
            self._current_iso_threshold = threshold

            self._apply_custom_lighting(params)
            self.plotter.add_axes()
            if reset_view:
                self.reset_camera()
            elif saved_camera_position is not None:
                # Restore previous camera state when switching from other modes
                try:
                    self.plotter.camera.position = saved_camera_position
                    self.plotter.camera.focal_point = saved_camera_focal_point
                    self.plotter.camera.up = saved_camera_view_up
                    self.plotter.camera.view_angle = saved_camera_view_angle
                    self.plotter.render()
                except Exception:
                    pass
        except Exception as e:
            print(f"Isosurface error: {e}")
            self.iso_actor = None

    def render_isosurface_auto(self, reset_view=True):
        """Render isosurface with current threshold parameter."""
        params = self._get_render_params()
        self.render_isosurface(threshold=params.get('threshold', 300), reset_view=reset_view)

    def _apply_custom_lighting(self, params):
        """Apply custom lighting configuration based on parameters."""
        if 'light_angle' in params and params['light_angle'] is not None:
            import math
            angle = params['light_angle']
            rad = math.radians(angle)
            light_pos = [10 * math.cos(rad), 10 * math.sin(rad), 10]
            self.plotter.remove_all_lights()
            self.plotter.add_light(pv.Light(position=light_pos, intensity=1.0))
