"""
Clip plane handler for volume visualization.
"""

from typing import Optional, Dict, Any
import pyvista as pv


class ClipHandler:
    """
    Handles clip plane logic for visualization.
    Designed for composition with GUI/rendering classes.
    """

    def __init__(self, plotter, clip_panel, render_engine=None):
        """
        Initialize clip handler.
        
        Args:
            plotter: BackgroundPlotter instance
            clip_panel: ClipPlanePanel for UI controls
            render_engine: Optional RenderEngine for rendering callbacks
        """
        self.plotter = plotter
        self.clip_panel = clip_panel
        self.render_engine = render_engine

    def update_clip_panel_state(self, active_mode: str):
        """Enable/Disable clip panel based on active mode."""
        if not self.clip_panel:
            return

        supported_modes = ['volume', 'mesh', 'iso']
        is_supported = active_mode in supported_modes

        self.clip_panel.setEnabled(is_supported)
        self.clip_panel.enable_checkbox.blockSignals(True)
        self.clip_panel.enable_checkbox.setChecked(False)
        self.clip_panel.enable_checkbox.blockSignals(False)
        self.clip_panel._enabled = False
        self.clip_panel._update_slider_state()

    def on_clip_toggled(self, enabled: bool):
        """Handle clip plane enable/disable toggle."""
        if enabled:
            self.apply_clip_planes()
        else:
            # Force full re-render to restore unclipped data
            if self.render_engine:
                mode = self.render_engine.active_view_mode
                if mode == 'volume':
                    self.render_engine.render_volume(reset_view=True)
                elif mode == 'slices':
                    self.render_engine.render_slices(reset_view=True)
                elif mode == 'iso':
                    self.render_engine.render_isosurface_auto(reset_view=True)
                elif mode == 'mesh':
                    self.render_engine.render_mesh(reset_view=True)

    def apply_clip_planes(self):
        """Apply clip planes to current visualization."""
        if not self.clip_panel:
            return

        clip_vals = self.clip_panel.get_clip_values()
        if not clip_vals['enabled']:
            return

        EPS = 0.005
        for axis in ['x', 'y', 'z']:
            if not clip_vals[f'invert_{axis}']:
                clip_vals[axis] = max(EPS, clip_vals[axis])
            else:
                clip_vals[axis] = min(1.0 - EPS, clip_vals[axis])

        try:
            if self.render_engine is None:
                return

            data_source = None
            mode = self.render_engine.active_view_mode

            if mode == 'volume':
                data_source = self.render_engine.grid
            elif mode == 'mesh':
                data_source = self.render_engine.mesh
            elif mode == 'iso':
                params = self.render_engine.params_panel.get_current_values() if self.render_engine.params_panel else {}
                thresh = params.get('threshold', 300)
                if thresh in self.render_engine._iso_cache:
                    data_source = self.render_engine._iso_cache[thresh]
                elif self.render_engine.grid:
                    data_source = self.render_engine.grid.cell_data_to_point_data().contour([thresh])

            if data_source is None:
                return

            bounds = data_source.bounds

            x_min = bounds[0]
            x_max = bounds[0] + (bounds[1] - bounds[0]) * clip_vals['x']
            y_min = bounds[2]
            y_max = bounds[2] + (bounds[3] - bounds[2]) * clip_vals['y']
            z_min = bounds[4]
            z_max = bounds[4] + (bounds[5] - bounds[4]) * clip_vals['z']

            if clip_vals['invert_x']:
                x_min, x_max = x_max, bounds[1]
            if clip_vals['invert_y']:
                y_min, y_max = y_max, bounds[3]
            if clip_vals['invert_z']:
                z_min, z_max = z_max, bounds[5]

            clip_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

            self.plotter.clear()
            self.plotter.add_axes()
            params = self.render_engine.params_panel.get_current_values() if self.render_engine.params_panel else {}

            if mode == 'volume' and self.render_engine.grid is not None:
                clipped = self.render_engine.grid.clip_box(clip_bounds, invert=False)
                if clipped.n_cells > 0:
                    self.plotter.add_mesh(
                        clipped,
                        scalars="values",
                        cmap=params.get('colormap', 'bone'),
                        clim=params.get('clim', [0, 1000]),
                        show_scalar_bar=True,
                        opacity=0.5
                    )

            elif mode == 'mesh' and self.render_engine.mesh is not None:
                clipped = self.render_engine.mesh.clip_box(clip_bounds, invert=False)
                if clipped.n_points > 0:
                    self.plotter.add_mesh(
                        clipped,
                        scalars="IsPore" if "IsPore" in clipped.array_names else None,
                        cmap=["gray", "red"] if "IsPore" in clipped.array_names else params.get('colormap', 'viridis'),
                        show_scalar_bar=False,
                        smooth_shading=True
                    )

            elif mode == 'iso':
                clipped = data_source.clip_box(clip_bounds, invert=False)
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

                coloring = params.get('coloring_mode', 'Solid Color')
                if coloring == 'Solid Color':
                    self.plotter.add_mesh(clipped, color=params.get('solid_color', 'ivory'), **mesh_kwargs)
                elif coloring == 'Depth (Z-Axis)':
                    clipped["Elevation"] = clipped.points[:, 2]
                    self.plotter.add_mesh(clipped, scalars="Elevation", cmap=params.get('colormap', 'viridis'), **mesh_kwargs)
                else:
                    self.plotter.add_mesh(clipped, color='white', **mesh_kwargs)

            self.render_engine._apply_custom_lighting(params)
            self.plotter.render()

        except Exception as e:
            print(f"[ClipHandler] Error: {e}")
            self.on_clip_toggled(False)
