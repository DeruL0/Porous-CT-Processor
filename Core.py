import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple

# ==========================================
# Core Data Structures and Interface Definitions
# ==========================================

@dataclass
class VolumeData:
    """
    Unified Data Transfer Object (DTO).
    Converts data from DICOM, NIfTI, or generation algorithms into this format.
    """
    raw_data: np.ndarray            # 3D Matrix (Z, Y, X)
    spacing: Tuple[float, float, float]  # Pixel spacing (z_spacing, y_spacing, x_spacing)
    origin: Tuple[float, float, float]   # Origin coordinates
    metadata: dict = field(default_factory=dict) # Extra metadata (e.g., PatientID, Modality)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return self.raw_data.shape

class BaseLoader(ABC):
    """Abstract base class for data loaders"""
    @abstractmethod
    def load(self, source: str) -> VolumeData:
        """Load data and return a VolumeData object"""
        pass

class BaseProcessor(ABC):
    """Abstract base class for image processors"""
    @abstractmethod
    def process(self, data: VolumeData, **kwargs) -> VolumeData:
        """Input data, execute algorithm, and return new processed data"""
        pass

class BaseVisualizer(ABC):
    """Abstract base class for visualizers"""
    @abstractmethod
    def set_data(self, data: VolumeData):
        """Inject data"""
        pass

    @abstractmethod
    def show(self):
        """Show default view"""
        pass