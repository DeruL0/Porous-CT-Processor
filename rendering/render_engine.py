"""
Core rendering engine for volumetric data visualization.
Provides reusable rendering methods independent of GUI framework.
Includes LOD (Level of Detail) support for large volume handling.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pyvista as pv
from core import VolumeData
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
        
        # GPU capability
        self.gpu_available, self.gpu_info = check_gpu_volume_rendering()
        
        # Actors
        self.volume_actor = None

    def update_status(self, message: str):
        """Update status via callback."""
        if self._status_callback:
            self._status_callback(message)
        else:
            print(f"[RenderEngine] {message}")

    def set_data(self, data: VolumeData):
        """Set volume data and prepare grid. Clears previous data first."""
        import gc
        
        # Clear previous data to free memory
        if self.data is not None or self.grid is not None:
            self.update_status("Clearing previous data...")
        
        self.data = None
        self.grid = None
        self.mesh = None
        self.volume_actor = None
        self._iso_cache = {}
        self._cached_vol_grid = None
        self._lod_pyramid = None
        
        # Force garbage collection to free memory before loading new data
        gc.collect()
        
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
        grid = pv.ImageData()
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def clear_view(self):
        """Clear all actors from plotter."""
        self.plotter.clear()
        self.plotter.add_axes()
        self.active_view_mode = None
        self.volume_actor = None
        if self.params_panel:
            self.params_panel.set_mode(None)

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

    def render_mesh(self, reset_view=True):
        """Render PNM mesh with PoreRadius colormap."""
        if not self.mesh:
            return
        if reset_view or self.active_view_mode != 'mesh':
            self.clear_view()
            self.active_view_mode = 'mesh'
            self._update_clip_panel_state()
            if self.params_panel:
                self.params_panel.set_mode('mesh')
            self.plotter.enable_lightkit()

        params = self.params_panel.get_current_values() if self.params_panel else {}

        if "PoreRadius" in self.mesh.point_data:
            self.plotter.add_mesh(
                self.mesh,
                scalars="PoreRadius",
                cmap=params.get('colormap', 'viridis'),
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
        """
        Optimized Volume Rendering.
        Updates Opacity/Color in-place when possible.
        """
        if not self.grid:
            return

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
            params = self.params_panel.get_current_values() if self.params_panel else {}

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
        else:
            # Fast update without clearing
            self.update_status("Updating volume properties...")
            params = self.params_panel.get_current_values() if self.params_panel else {}
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

        if reset_view or self.active_view_mode != 'slices':
            self.clear_view()
            self.active_view_mode = 'slices'
            self._update_clip_panel_state()
            if self.params_panel:
                self.params_panel.set_mode('slices')
            self.plotter.enable_lightkit()
            self.plotter.show_grid()

        params = self.params_panel.get_current_values() if self.params_panel else {}
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

    def render_isosurface(self, threshold=300, reset_view=True):
        """Render isosurface at specified threshold."""
        if not self.grid:
            return

        self.update_status(f"Generating isosurface ({threshold})...")
        self.clear_view()
        self.active_view_mode = 'iso'
        self._update_clip_panel_state()
        if self.params_panel:
            self.params_panel.set_mode('iso')
        self.plotter.enable_lightkit()
        params = self.params_panel.get_current_values() if self.params_panel else {}

        try:
            # Check cache
            if threshold in self._iso_cache:
                contours = self._iso_cache[threshold]
            else:
                # Use potentially downsampled grid for large volumes
                safe_grid = self._get_grid_for_isosurface()
                if safe_grid is None:
                    self.update_status("No grid available for isosurface")
                    return
                
                grid_points = safe_grid.cell_data_to_point_data()
                contours = grid_points.contour(isosurfaces=[threshold])
                contours.compute_normals(inplace=True)
                self._iso_cache[threshold] = contours

            style_map = {'Surface': 'surface', 'Wireframe': 'wireframe', 'Wireframe + Surface': 'surface'}
            render_style = style_map.get(params.get('render_style', 'Surface'), 'surface')
            show_edges = params.get('render_style') == 'Wireframe + Surface'

            mesh_kwargs = {
                'style': render_style,
                'show_edges': show_edges,
                'smooth_shading': True,
                'specular': 0.4,
                'diffuse': 0.7,
                'ambient': 0.15,
                'lighting': True
            }

            mode = params.get('coloring_mode', 'Solid Color')
            if mode == 'Solid Color':
                self.plotter.add_mesh(contours, color=params.get('solid_color', 'ivory'), **mesh_kwargs)
            elif mode == 'Depth (Z-Axis)':
                contours["Elevation"] = contours.points[:, 2]
                self.plotter.add_mesh(contours, scalars="Elevation", cmap=params.get('colormap', 'viridis'), **mesh_kwargs)
            elif mode == 'Radial (Center Dist)':
                dist = np.linalg.norm(contours.points - contours.center, axis=1)
                contours["RadialDistance"] = dist
                self.plotter.add_mesh(contours, scalars="RadialDistance", cmap=params.get('colormap', 'viridis'), **mesh_kwargs)

            self._apply_custom_lighting(params)
            self.plotter.add_axes()
            if reset_view:
                self.reset_camera()
        except Exception as e:
            print(f"Isosurface error: {e}")

    def render_isosurface_auto(self, reset_view=True):
        """Render isosurface with current threshold parameter."""
        params = self.params_panel.get_current_values() if self.params_panel else {}
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
