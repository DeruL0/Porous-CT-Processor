"""
Level of Detail (LOD) manager for efficient volume rendering.
Provides multi-resolution pyramid and GPU capability detection.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import pyvista as pv

# Try to import VTK for GPU detection
try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


def check_gpu_volume_rendering() -> Tuple[bool, str]:
    """
    Check if GPU volume rendering is available.
    
    Returns:
        Tuple of (is_available, description)
    """
    if not VTK_AVAILABLE:
        return False, "VTK not available"
    
    try:
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.Render()
        
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        vol_property = vtk.vtkVolumeProperty()
        
        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)
        
        if mapper.IsRenderSupported(render_window, vol_property):
            return True, "GPU volume rendering supported"
        else:
            return False, "GPU volume rendering not supported by hardware"
    except Exception as e:
        return False, f"GPU check failed: {e}"


class LODPyramid:
    """
    Multi-resolution pyramid for efficient rendering.
    Creates downsampled versions of volume data at multiple levels.
    """
    
    def __init__(self, grid: pv.ImageData, levels: int = 3):
        """
        Create LOD pyramid from a PyVista ImageData grid.
        
        Args:
            grid: Original full-resolution grid.
            levels: Number of LOD levels to create (including original).
        """
        self.levels: List[pv.ImageData] = []
        self.level_info: List[Dict] = []
        
        self._build_pyramid(grid, levels)
    
    def _build_pyramid(self, grid: pv.ImageData, levels: int):
        """Build the LOD pyramid."""
        current = grid
        
        for level in range(levels):
            self.levels.append(current)
            self.level_info.append({
                'level': level,
                'n_cells': current.n_cells,
                'dimensions': current.dimensions,
                'memory_mb': current.n_cells * 4 / (1024 * 1024)  # Approximate float32
            })
            
            if level < levels - 1:
                # Create downsampled version
                dims = current.dimensions
                if min(dims) > 10:  # Ensure we have enough resolution
                    try:
                        current = current.extract_subset(
                            voi=(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1),
                            rate=(2, 2, 2)
                        )
                    except:
                        break  # Can't downsample further
                else:
                    break
    
    def get_level(self, level: int) -> Optional[pv.ImageData]:
        """Get grid at specified LOD level."""
        if 0 <= level < len(self.levels):
            return self.levels[level]
        return self.levels[-1] if self.levels else None
    
    def get_for_distance(self, camera_distance: float, 
                         far_threshold: float = 1000,
                         mid_threshold: float = 500) -> pv.ImageData:
        """
        Get appropriate LOD based on camera distance.
        
        Args:
            camera_distance: Distance from camera to object.
            far_threshold: Distance above which to use lowest LOD.
            mid_threshold: Distance above which to use medium LOD.
            
        Returns:
            Grid at appropriate LOD level.
        """
        if camera_distance > far_threshold and len(self.levels) > 2:
            return self.levels[2]
        elif camera_distance > mid_threshold and len(self.levels) > 1:
            return self.levels[1]
        return self.levels[0]
    
    def get_for_memory(self, max_memory_mb: float = 400) -> pv.ImageData:
        """
        Get highest resolution LOD that fits in memory budget.
        
        Args:
            max_memory_mb: Maximum memory budget in MB.
            
        Returns:
            Grid that fits in memory budget.
        """
        for i, info in enumerate(self.level_info):
            if info['memory_mb'] <= max_memory_mb:
                return self.levels[i]
        return self.levels[-1] if self.levels else None
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    def __repr__(self):
        info_str = ", ".join([f"L{i['level']}:{i['n_cells']:,}" for i in self.level_info])
        return f"LODPyramid({info_str})"


class LODRenderManager:
    """
    Manages LOD-based rendering with automatic level selection.
    """
    
    def __init__(self, plotter, status_callback=None):
        """
        Args:
            plotter: PyVista plotter instance.
            status_callback: Optional status update callback.
        """
        self.plotter = plotter
        self._status_callback = status_callback
        self.pyramid: Optional[LODPyramid] = None
        self.current_level: int = 0
        
        # Check GPU capability once
        self.gpu_available, self.gpu_info = check_gpu_volume_rendering()
        if status_callback:
            status_callback(f"GPU rendering: {self.gpu_info}")
    
    def set_grid(self, grid: pv.ImageData, levels: int = 3):
        """
        Create LOD pyramid from grid.
        
        Args:
            grid: Original grid.
            levels: Number of LOD levels.
        """
        self.pyramid = LODPyramid(grid, levels)
        if self._status_callback:
            self._status_callback(f"Created LOD pyramid: {self.pyramid}")
    
    def get_render_grid(self, max_memory_mb: float = 400) -> Optional[pv.ImageData]:
        """Get grid suitable for rendering within memory budget."""
        if self.pyramid is None:
            return None
        return self.pyramid.get_for_memory(max_memory_mb)
    
    def update_lod_for_camera(self):
        """Update LOD based on current camera position."""
        if self.pyramid is None or len(self.pyramid.levels) == 0:
            return
        
        # Get camera distance
        camera = self.plotter.camera
        if camera and hasattr(camera, 'distance'):
            distance = camera.distance
            grid = self.pyramid.get_for_distance(distance)
            # Here you would trigger a re-render with the new LOD
            # Implementation depends on how rendering is structured
