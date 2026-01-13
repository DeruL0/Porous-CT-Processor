"""
ROI (Region of Interest) handler for volume visualization.
Handles all ROI selection, preview rendering, and extraction logic.
"""

from typing import Optional, Callable, Tuple
import numpy as np
import pyvista as pv
from core import VolumeData


class ROIHandler:
    """
    Handles ROI selection, preview rendering, and extraction logic.
    Designed for composition with GUI/rendering classes.
    """

    # Shape color mapping
    SHAPE_COLORS = {'box': 'cyan', 'sphere': 'magenta', 'cylinder': 'yellow'}

    def __init__(self, plotter, roi_panel, data_manager=None, status_callback=None):
        """
        Initialize ROI handler.
        
        Args:
            plotter: BackgroundPlotter instance
            roi_panel: ROIPanel for UI controls
            data_manager: Optional DataManager for data flow
            status_callback: Optional callback for status updates
        """
        self.plotter = plotter
        self.roi_panel = roi_panel
        self._data_manager = data_manager
        self._status_callback = status_callback
        
        # Data references (set externally)
        self.data: Optional[VolumeData] = None
        self.grid = None
        
        # Widget state
        self._box_widget = None
        self._transform: Optional[np.ndarray] = None
        self._base_size: Optional[Tuple[float, float, float]] = None
        self._current_size: Optional[Tuple[float, float, float]] = None
        
        # Preview rendering state
        self._shape_preview = None
        self._preview_renderer = None

    def update_status(self, message: str):
        """Update status via callback."""
        if self._status_callback:
            self._status_callback(message)

    def set_data(self, data: VolumeData, grid):
        """Set data reference for ROI extraction."""
        self.data = data
        self.grid = grid

    # ==========================================
    # Widget Management
    # ==========================================

    def on_shape_changed(self, shape: str):
        """Handle shape change - refresh widget with new color and update preview."""
        if not self.roi_panel.enable_checkbox.isChecked():
            self.update_status(f"ROI shape set to: {shape}")
            return
        
        # Preserve current bounds before clearing
        current_bounds = self.roi_panel.get_bounds()
        if current_bounds is None and self.grid is not None:
            current_bounds = self.grid.bounds
        
        self.clear_widgets()
        
        if current_bounds is None or self.grid is None:
            return
        
        # Re-create widget with new shape's color
        self._create_box_widget(current_bounds, shape)
        self._update_preview(current_bounds)
        
        status_map = {
            'box': "ROI Box: Drag to resize, rotate with edges",
            'sphere': "ROI Ellipsoid: Inscribed in box, inherits rotation",
            'cylinder': "ROI Cylinder: Inscribed in box, inherits rotation"
        }
        self.update_status(status_map.get(shape, f"ROI shape: {shape}"))
        self.plotter.render()

    def on_toggled(self, enabled: bool):
        """Handle ROI mode toggle."""
        if not enabled:
            self.clear_widgets()
            self.roi_panel.update_bounds(None)
            self.update_status("ROI mode disabled")
            return

        if self.grid is None:
            self.roi_panel.enable_checkbox.setChecked(False)
            return

        shape = self.roi_panel.get_shape()
        bounds = self.grid.bounds
        
        self._create_box_widget(bounds, shape)
        
        status_map = {
            'box': "ROI Box: Drag to resize, rotate with edges",
            'sphere': "ROI Ellipsoid: Drag box - ellipsoid fills box, can rotate",
            'cylinder': "ROI Cylinder: Drag box - elliptical cylinder, can rotate"
        }
        self.update_status(status_map.get(shape, "ROI: Drag to select region"))

    def _create_box_widget(self, bounds: tuple, shape: str):
        """Create box widget for ROI selection."""
        color = self.SHAPE_COLORS.get(shape, 'cyan')
        
        self.plotter.add_box_widget(
            callback=self._on_bounds_changed,
            bounds=bounds,
            factor=1.0,
            rotation_enabled=True,
            color=color,
            use_planes=False
        )
        
        # Store references
        if self.plotter.box_widgets:
            self._box_widget = self.plotter.box_widgets[-1]
        else:
            self._box_widget = None
        
        # Store initial size for rotation tracking
        initial_size = (
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        self._base_size = initial_size
        self._current_size = initial_size

    def _on_bounds_changed(self, bounds):
        """Callback when box widget is moved/rotated/scaled."""
        if hasattr(bounds, 'bounds'):
            actual_bounds = bounds.bounds
        else:
            actual_bounds = bounds
        
        # Capture transform matrix
        self._transform = None
        if self._box_widget:
            try:
                import vtk
                transform = vtk.vtkTransform()
                self._box_widget.GetTransform(transform)
                
                matrix = transform.GetMatrix()
                self._transform = np.array([
                    [matrix.GetElement(i, j) for j in range(4)]
                    for i in range(4)
                ])
                
                # Update size based on scale in transform
                scale_x = np.linalg.norm(self._transform[:3, 0])
                scale_y = np.linalg.norm(self._transform[:3, 1])
                scale_z = np.linalg.norm(self._transform[:3, 2])
                
                if self._base_size:
                    self._current_size = (
                        self._base_size[0] * scale_x,
                        self._base_size[1] * scale_y,
                        self._base_size[2] * scale_z
                    )
            except Exception as e:
                print(f"[ROI] Transform extraction failed: {e}")
        
        self.roi_panel.update_bounds(actual_bounds)
        self._update_preview(actual_bounds)

    def clear_widgets(self):
        """Clear all ROI-related widgets and preview shapes."""
        self.plotter.clear_box_widgets()
        self.plotter.clear_sphere_widgets()
        self._box_widget = None
        
        # Remove preview
        if self._shape_preview:
            if self._preview_renderer:
                self._preview_renderer.RemoveActor(self._shape_preview)
            self.plotter.remove_actor(self._shape_preview)
            self._shape_preview = None

    # ==========================================
    # Preview Rendering
    # ==========================================

    def _setup_preview_renderer(self):
        """Setup foreground renderer layer for ROI shape preview."""
        if self._preview_renderer:
            return self._preview_renderer
        
        import vtk
        
        self._preview_renderer = vtk.vtkRenderer()
        self._preview_renderer.SetLayer(1)
        self._preview_renderer.InteractiveOff()
        self._preview_renderer.SetBackground(0, 0, 0)
        self._preview_renderer.SetBackgroundAlpha(0.0)
        
        render_window = self.plotter.ren_win
        render_window.SetNumberOfLayers(2)
        render_window.AddRenderer(self._preview_renderer)
        
        main_camera = self.plotter.renderer.GetActiveCamera()
        self._preview_renderer.SetActiveCamera(main_camera)
        
        return self._preview_renderer

    def _update_preview(self, bounds):
        """Update inscribed shape preview with box widget rotation."""
        if bounds is None:
            return
        
        shape = self.roi_panel.get_shape()
        
        # Remove old preview
        if self._shape_preview:
            if self._preview_renderer:
                self._preview_renderer.RemoveActor(self._shape_preview)
            self.plotter.remove_actor(self._shape_preview)
            self._shape_preview = None
        
        if shape == 'box':
            return
        
        # Use stored size (not AABB which expands on rotation)
        if self._current_size:
            size_x, size_y, size_z = self._current_size
        else:
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
        
        # Create mesh at origin
        if shape == 'sphere':
            mesh = pv.ParametricEllipsoid(size_x / 2, size_y / 2, size_z / 2)
            preview_color = 'magenta'
        elif shape == 'cylinder':
            mesh = self._create_cylinder_mesh(size_x, size_y, size_z)
            preview_color = 'yellow'
        else:
            return
        
        # Apply pure rotation + translation
        # Calculate center from AABB bounds (this is always the visual center)
        center = (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        )
        
        if self._transform is not None:
            # Extract pure rotation (normalize columns to remove scale)
            rotation = self._transform[:3, :3].copy()
            for i in range(3):
                col_norm = np.linalg.norm(rotation[:, i])
                if col_norm > 1e-6:
                    rotation[:, i] /= col_norm
            
            # Apply rotation first, then translate to AABB center
            pure_transform = np.eye(4)
            pure_transform[:3, :3] = rotation
            mesh.transform(pure_transform, inplace=True)
            mesh.translate(center, inplace=True)
        else:
            mesh.translate(center, inplace=True)
        
        # Add preview mesh
        self._shape_preview = self.plotter.add_mesh(
            mesh,
            color=preview_color,
            style='wireframe',
            line_width=2,
            opacity=1.0,
            name='roi_shape_preview'
        )
        
        # Move to foreground layer
        if self._shape_preview:
            renderer = self._setup_preview_renderer()
            if self.plotter.renderer.HasViewProp(self._shape_preview):
                self.plotter.renderer.RemoveActor(self._shape_preview)
            renderer.AddActor(self._shape_preview)

    def _create_cylinder_mesh(self, size_x: float, size_y: float, size_z: float):
        """Create elliptical cylinder mesh centered at origin."""
        ry, rz = size_y / 2, size_z / 2
        n_pts = 32
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        
        points_bottom = np.column_stack([
            np.full(n_pts, -size_x / 2),
            ry * np.cos(angles),
            rz * np.sin(angles)
        ])
        points_top = np.column_stack([
            np.full(n_pts, size_x / 2),
            ry * np.cos(angles),
            rz * np.sin(angles)
        ])
        
        all_points = np.vstack([points_bottom, points_top])
        
        faces = []
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            faces.append([3, i, next_i, n_pts + i])
            faces.append([3, next_i, n_pts + next_i, n_pts + i])
        faces_flat = np.hstack([[item for f in faces for item in f]])
        
        return pv.PolyData(all_points, faces_flat)

    # ==========================================
    # ROI Application
    # ==========================================

    def on_apply(self, set_data_callback: Callable[[VolumeData], None]):
        """Apply ROI extraction and update data."""
        if self.data is None:
            return

        shape = self.roi_panel.get_shape()
        roi_bounds = self.roi_panel.get_bounds()
        
        if roi_bounds is None:
            return
        
        try:
            if shape == 'box':
                extracted = self._extract_box(roi_bounds)
            elif shape == 'sphere':
                extracted = self._extract_ellipsoid(roi_bounds)
            elif shape == 'cylinder':
                extracted = self._extract_cylinder(roi_bounds)
            else:
                extracted = None
            
            if extracted is not None:
                if self._data_manager:
                    self._data_manager.set_roi_data(extracted)
                set_data_callback(extracted)
                self.update_status(f"ROI applied ({shape}): {extracted.raw_data.shape}")
                self.roi_panel.enable_checkbox.setChecked(False)
                self.clear_widgets()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"ROI extraction failed: {e}")

    def on_reset(self, set_data_callback: Callable[[VolumeData], None]):
        """Reset ROI and restore original data."""
        self.clear_widgets()
        if self._data_manager:
            self._data_manager.clear_roi()
            if self._data_manager.raw_ct_data:
                set_data_callback(self._data_manager.raw_ct_data)
        self.update_status("ROI reset")

    # ==========================================
    # Extraction Methods
    # ==========================================

    def _bounds_to_voxel_indices(self, bounds: tuple) -> tuple:
        """Convert world bounds to voxel indices."""
        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin
        
        i_start = max(0, int((bounds[0] - origin[0]) / spacing[0]))
        i_end = min(raw.shape[0], int((bounds[1] - origin[0]) / spacing[0]))
        j_start = max(0, int((bounds[2] - origin[1]) / spacing[1]))
        j_end = min(raw.shape[1], int((bounds[3] - origin[1]) / spacing[1]))
        k_start = max(0, int((bounds[4] - origin[2]) / spacing[2]))
        k_end = min(raw.shape[2], int((bounds[5] - origin[2]) / spacing[2]))
        
        return i_start, i_end, j_start, j_end, k_start, k_end

    def _extract_box(self, bounds: tuple) -> Optional[VolumeData]:
        """Extract box-shaped sub-volume."""
        if self.data is None or self.data.raw_data is None:
            return None

        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin
        
        i_start, i_end, j_start, j_end, k_start, k_end = self._bounds_to_voxel_indices(bounds)
        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
        
        if sub_data.size == 0:
            return None

        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )
        
        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Box ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds

        return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=new_metadata)

    def _extract_ellipsoid(self, bounds: tuple) -> Optional[VolumeData]:
        """Extract ellipsoid sub-volume inscribed in box bounds."""
        if self.data is None or self.data.raw_data is None:
            return None

        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin
        
        i_start, i_end, j_start, j_end, k_start, k_end = self._bounds_to_voxel_indices(bounds)
        
        # Calculate center and radii
        rx = (bounds[1] - bounds[0]) / 2
        ry = (bounds[3] - bounds[2]) / 2
        rz = (bounds[5] - bounds[4]) / 2
        
        ci = ((bounds[0] + bounds[1]) / 2 - origin[0]) / spacing[0]
        cj = ((bounds[2] + bounds[3]) / 2 - origin[1]) / spacing[1]
        ck = ((bounds[4] + bounds[5]) / 2 - origin[2]) / spacing[2]
        
        # Create ellipsoid mask
        ii, jj, kk = np.meshgrid(
            np.arange(i_start, i_end),
            np.arange(j_start, j_end),
            np.arange(k_start, k_end),
            indexing='ij'
        )
        
        dist = np.sqrt(
            ((ii - ci) * spacing[0] / rx) ** 2 +
            ((jj - cj) * spacing[1] / ry) ** 2 +
            ((kk - ck) * spacing[2] / rz) ** 2
        )
        mask = dist <= 1.0
        
        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
        sub_data[~mask] = sub_data.min()
        
        if sub_data.size == 0:
            return None

        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )
        
        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Ellipsoid ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds
        new_metadata['ROI_Radii'] = (rx, ry, rz)

        return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=new_metadata)

    def _extract_cylinder(self, bounds: tuple) -> Optional[VolumeData]:
        """Extract elliptical cylinder inscribed in box bounds (supports rotation)."""
        if self.data is None or self.data.raw_data is None:
            return None

        raw = self.data.raw_data
        spacing = self.data.spacing
        origin = self.data.origin
        
        i_start, i_end, j_start, j_end, k_start, k_end = self._bounds_to_voxel_indices(bounds)
        
        # Use stored size if available (rotation-invariant)
        if self._current_size:
            size_x, size_y, size_z = self._current_size
        else:
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
        
        # Radii for elliptical cross-section
        rx = size_x / 2  # Cylinder length axis
        ry = size_y / 2
        rz = size_z / 2
        
        # Center in world coordinates (from AABB)
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
        
        # Create voxel coordinate grids
        ii, jj, kk = np.meshgrid(
            np.arange(i_start, i_end),
            np.arange(j_start, j_end),
            np.arange(k_start, k_end),
            indexing='ij'
        )
        
        # Convert to world coordinates
        world_x = origin[0] + ii * spacing[0]
        world_y = origin[1] + jj * spacing[1]
        world_z = origin[2] + kk * spacing[2]
        
        # Get relative position to center
        rel_x = world_x - center[0]
        rel_y = world_y - center[1]
        rel_z = world_z - center[2]
        
        # Apply inverse rotation if transform exists
        if self._transform is not None:
            # Extract rotation matrix and normalize (remove scale)
            rotation = self._transform[:3, :3].copy()
            for i in range(3):
                col_norm = np.linalg.norm(rotation[:, i])
                if col_norm > 1e-6:
                    rotation[:, i] /= col_norm
            
            # Inverse rotation = transpose for orthogonal matrix
            inv_rotation = rotation.T
            
            # Transform relative coordinates to local space
            local_x = inv_rotation[0, 0] * rel_x + inv_rotation[0, 1] * rel_y + inv_rotation[0, 2] * rel_z
            local_y = inv_rotation[1, 0] * rel_x + inv_rotation[1, 1] * rel_y + inv_rotation[1, 2] * rel_z
            local_z = inv_rotation[2, 0] * rel_x + inv_rotation[2, 1] * rel_y + inv_rotation[2, 2] * rel_z
        else:
            local_x = rel_x
            local_y = rel_y
            local_z = rel_z
        
        # Cylinder mask: inside if (y/ry)^2 + (z/rz)^2 <= 1 AND abs(x) <= rx
        dist_2d = np.sqrt((local_y / ry) ** 2 + (local_z / rz) ** 2)
        mask = (dist_2d <= 1.0) & (np.abs(local_x) <= rx)
        
        sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
        sub_data[~mask] = sub_data.min()
        
        if sub_data.size == 0:
            return None

        new_origin = (
            origin[0] + i_start * spacing[0],
            origin[1] + j_start * spacing[1],
            origin[2] + k_start * spacing[2]
        )
        
        new_metadata = dict(self.data.metadata)
        new_metadata['Type'] = f"ROI Elliptical Cylinder ({sub_data.shape})"
        new_metadata['ROI_Bounds'] = bounds
        new_metadata['ROI_Radii_YZ'] = (ry, rz)
        if self._transform is not None:
            new_metadata['ROI_Rotated'] = True

        return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=new_metadata)
