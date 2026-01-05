import numpy as np
import pyvista as pv
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Callable


# ==========================================
# Core Data Structures and Interface Definitions
# ==========================================

@dataclass
class VolumeData:
    """
    Unified Data Transfer Object (DTO).
    Now supports both Voxel Grid (numpy) and Mesh (PyVista PolyData).

    Attributes:
        raw_data (Optional[np.ndarray]): 3D Matrix (Z, Y, X) for voxel data.
        mesh (Optional[pv.PolyData]): 3D Mesh for PNM or Surface data.
        spacing (Tuple): Voxel spacing (z, y, x) in mm.
        origin (Tuple): Origin coordinates (z, y, x) in mm.
        metadata (Dict): Arbitrary metadata (SampleID, etc.).
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
        return self.mesh is not None


class BaseLoader(ABC):
    """Abstract base class for data acquisition strategies."""

    @abstractmethod
    def load(self, source: str) -> VolumeData:
        pass


class BaseProcessor(ABC):
    """Abstract base class for volumetric processing algorithms."""

    @abstractmethod
    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None, **kwargs) -> VolumeData:
        """
        Args:
            data: Input VolumeData
            callback: Optional function (progress_percent, status_message) -> None
            **kwargs: Algorithm specific parameters
        """
        pass


class BaseVisualizer(ABC):
    """Abstract base class for visualization controllers."""

    @abstractmethod
    def set_data(self, data: VolumeData) -> None:
        pass

    @abstractmethod
    def show(self) -> None:
        pass