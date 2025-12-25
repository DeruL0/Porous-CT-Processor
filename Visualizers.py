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
        grid.dimensions = np.array(self.data.raw_data.shape) + 1
        grid.origin = self.data.origin
        grid.spacing = self.data.spacing
        grid.cell_data["values"] = self.data.raw_data.flatten(order="F")
        self.grid = grid

    def show(self):
        self.render_volume()

    def render_volume(self, cmap="bone", opacity="sigmoid"):
        if not self.grid: raise ValueError("No data, please call set_data() first")
        print("[Visualizer] Starting volume rendering...")
        p = pv.Plotter()
        p.add_text(f"Volume Rendering\n{self.data.metadata}", font_size=10)
        p.add_volume(self.grid, cmap=cmap, opacity=opacity, shade=True)
        p.add_axes()
        p.show()

    def render_slices(self):
        if not self.grid: raise ValueError("No data")
        print("[Visualizer] Starting slice view...")
        p = pv.Plotter()
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

        contours = self.grid.contour(isosurfaces=[threshold])

        p = pv.Plotter()
        p.add_text(f"Isosurface > {threshold} HU\nColor: {color}", font_size=10)
        p.add_mesh(contours, color=color, opacity=opacity, smooth_shading=True)
        p.add_axes()
        p.show()