"""
Core data structures and abstract base classes.
"""

import numpy as np
import pyvista as pv
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Callable


@dataclass
class VolumeData:
    """
    Unified Data Transfer Object (DTO) for volumetric and mesh data.
    
    Attributes:
        raw_data (Optional[np.ndarray]): 3D Matrix (Z, Y, X) for voxel data.
        mesh (Optional[pv.PolyData]): 3D Mesh for PNM or Surface data.
        spacing (Tuple[float, float, float]): Voxel spacing (x, y, z) in mm.
        origin (Tuple[float, float, float]): Origin coordinates (x, y, z) in mm.
        metadata (Dict[str, Any]): Arbitrary metadata (SampleID, etc.).
    """
    raw_data: Optional[np.ndarray] = None
    mesh: Optional[pv.PolyData] = None
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Returns the shape of the volume (Z, Y, X) if raw_data exists."""
        if self.raw_data is not None:
            return self.raw_data.shape
        return (0, 0, 0)

    @property
    def has_mesh(self) -> bool:
        """Check if mesh data is present."""
        return self.mesh is not None


class BaseLoader(ABC):
    """Abstract base class for data acquisition strategies."""

    @abstractmethod
    def load(self, source: str, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        """
        Load data from a source path.
        
        Args:
            source (str): Path to file or directory.
            callback: Optional progress callback (percent, message).
            
        Returns:
            VolumeData: Loaded data object.
        """
        pass


class BaseProcessor(ABC):
    """Abstract base class for volumetric processing algorithms."""

    @abstractmethod
    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None, **kwargs) -> VolumeData:
        """
        Process volume data.
        
        Args:
            data (VolumeData): Input data.
            callback (Optional[Callable]): Progress callback (percent, message).
            **kwargs: Algorithm specific parameters.
            
        Returns:
            VolumeData: Processed result.
        """
        pass


class BaseVisualizer(ABC):
    """Abstract base class for visualization controllers."""

    @abstractmethod
    def set_data(self, data: VolumeData) -> None:
        """Set valid data to the visualizer."""
        pass

    @abstractmethod
    def show(self) -> None:
        """Show the visualization window."""
        pass
