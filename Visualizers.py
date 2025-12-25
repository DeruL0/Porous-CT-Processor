import numpy as np
import pyvista as pv
from typing import Optional
from Core import BaseVisualizer, VolumeData


# ==========================================
# Visualizer Implementation
# ==========================================

class PyVistaVisualizer(BaseVisualizer):
    """PyVista-based visualization implementation"""

    def __init__(self):
        self.grid = None
        self.data: Optional[VolumeData] = None

    def set_data(self, data: VolumeData):
        self.data = data
        self._create_grid()

    def _create_grid(self):
        if not self.data: return
        grid = pv.ImageData()
        # Create a grid where data represents voxels (cells)
        # Dimensions are shape + 1 because cells are defined by their corners
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def show(self):
        self.render_volume()

    def _create_plotter(self):
        """Helper to create a standard plotter"""
        p = pv.Plotter()
        # Removed SSAA to ensure compatibility and performance
        return p

    def render_volume(self, cmap="bone", opacity="sigmoid"):
        if not self.grid: raise ValueError("No data, please call set_data() first")
        print("[Visualizer] Starting volume rendering...")

        p = self._create_plotter()
        p.add_text(f"Volume Rendering\n{self.data.metadata}", font_size=10)

        # Convert cell data to point data for better volume rendering compatibility
        # This ensures the volume renders correctly without needing specific GPU mappers
        vol_grid = self.grid.cell_data_to_point_data()

        # Removed 'mapper="smart"' and 'shade=True' for a raw, clear view
        p.add_volume(vol_grid, cmap=cmap, opacity=opacity, shade=False)
        p.add_axes()
        p.show()

    def render_slices(self):
        if not self.grid: raise ValueError("No data")
        print("[Visualizer] Starting slice view...")

        p = self._create_plotter()
        p.add_text("Orthogonal Slices", font_size=10)
        slices = self.grid.slice_orthogonal()
        p.add_mesh(slices, cmap="bone")
        p.show_grid()
        p.show()

    def render_isosurface(self, threshold=300, color="ivory", opacity=1.0):
        if not self.grid: raise ValueError("No data")
        print(f"[Visualizer] Extracting isosurface (Threshold: {threshold})...")

        min_val, max_val = self.data.raw_data.min(), self.data.raw_data.max()
        if threshold < min_val or threshold > max_val:
            print(f"Warning: Threshold {threshold} is out of data range [{min_val}, {max_val}]")
            return

        # FIX: The contour filter requires Point Data, but we have Cell Data.
        # We must convert cell_data to point_data (interpolation) before contouring.
        print("[Visualizer] Converting cell data to point data for contouring...")
        grid_points = self.grid.cell_data_to_point_data()

        contours = grid_points.contour(isosurfaces=[threshold])

        p = self._create_plotter()
        p.add_text(f"Isosurface > {threshold} HU\nColor: {color}", font_size=10)

        # Removed PBR and metallic settings for a standard matte look
        p.add_mesh(contours, color=color, opacity=opacity, smooth_shading=True)
        p.add_axes()
        p.show()