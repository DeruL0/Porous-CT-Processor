"""
Time series data structures for 4D CT pore network tracking.

Provides data classes for tracking pore network changes across
multiple timepoints in a compression sequence.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class PoreStatus(Enum):
    """Status of a tracked pore across timepoints."""
    ACTIVE = "active"           # Pore is present and measurable
    COMPRESSED = "compressed"   # Pore volume has dropped below threshold
    UNKNOWN = "unknown"         # Initial state before tracking


@dataclass
class PNMSnapshot:
    """
    A single timepoint snapshot of the pore network model.
    
    This captures the state of all pores at a specific time index,
    including their spatial positions, sizes, and connectivity.
    
    Attributes:
        time_index: Integer index of this timepoint (0 = reference)
        pore_centers: Coordinates of pore centroids (N, 3) in world space
        pore_radii: Equivalent sphere radii for each pore (N,)
        pore_ids: Unique identifier for each pore (N,)
        pore_volumes: Voxel count for each pore (N,)
        connections: List of (pore_id_a, pore_id_b) tuples for throats
        segmented_regions: 3D label array from watershed (optional, memory intensive)
        metadata: Additional information (threshold used, etc.)
    """
    time_index: int
    pore_centers: np.ndarray
    pore_radii: np.ndarray
    pore_ids: np.ndarray
    pore_volumes: np.ndarray
    connections: List[Tuple[int, int]]
    segmented_regions: Optional[np.ndarray] = None
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_pores(self) -> int:
        """Number of pores in this snapshot."""
        return len(self.pore_ids)
    
    @property
    def num_connections(self) -> int:
        """Number of throat connections."""
        return len(self.connections)
    
    def get_pore_data(self, pore_id: int) -> Optional[Dict[str, Any]]:
        """Get all data for a specific pore by ID."""
        mask = self.pore_ids == pore_id
        if not np.any(mask):
            return None
        idx = np.argmax(mask)
        return {
            'id': int(pore_id),
            'center': self.pore_centers[idx].tolist(),
            'radius': float(self.pore_radii[idx]),
            'volume': float(self.pore_volumes[idx])
        }


@dataclass
class PoreTrackingResult:
    """
    Results of tracking pores across all timepoints.
    
    This maintains the correspondence between pores in the reference
    (t=0) and all subsequent timepoints, tracking volume changes
    and compression events.
    
    Attributes:
        reference_ids: List of pore IDs from the reference snapshot
        volume_history: {pore_id: [v0, v1, v2, ...]} volume at each timepoint
        status_history: {pore_id: [status0, status1, ...]} status at each timepoint
        id_mapping: {time_index: {ref_id: current_id}} maps reference to current IDs
        iou_history: {pore_id: [iou0, iou1, ...]} IoU scores for matching quality
    """
    reference_ids: List[int] = field(default_factory=list)
    volume_history: Dict[int, List[float]] = field(default_factory=dict)
    status_history: Dict[int, List[PoreStatus]] = field(default_factory=dict)
    id_mapping: Dict[int, Dict[int, int]] = field(default_factory=dict)
    iou_history: Dict[int, List[float]] = field(default_factory=dict)
    
    def get_volume_series(self, pore_id: int) -> List[float]:
        """Get volume history for a specific pore."""
        return self.volume_history.get(pore_id, [])
    
    def get_status_series(self, pore_id: int) -> List[PoreStatus]:
        """Get status history for a specific pore."""
        return self.status_history.get(pore_id, [])
    
    def get_compression_timepoint(self, pore_id: int) -> Optional[int]:
        """
        Get the timepoint at which a pore first became compressed.
        Returns None if the pore was never compressed.
        """
        statuses = self.status_history.get(pore_id, [])
        for i, status in enumerate(statuses):
            if status == PoreStatus.COMPRESSED:
                return i
        return None
    
    def get_compressed_pore_ids(self, at_timepoint: Optional[int] = None) -> List[int]:
        """
        Get list of pore IDs that are compressed.
        
        Args:
            at_timepoint: If specified, return pores compressed at that timepoint.
                         If None, return all pores that are currently compressed.
        """
        compressed = []
        for pore_id, statuses in self.status_history.items():
            if at_timepoint is not None:
                if at_timepoint < len(statuses) and statuses[at_timepoint] == PoreStatus.COMPRESSED:
                    compressed.append(pore_id)
            else:
                # Check if currently (last status) is compressed
                if statuses and statuses[-1] == PoreStatus.COMPRESSED:
                    compressed.append(pore_id)
        return compressed
    
    def get_active_pore_ids(self, at_timepoint: Optional[int] = None) -> List[int]:
        """Get list of pore IDs that are still active (not compressed)."""
        active = []
        for pore_id, statuses in self.status_history.items():
            if at_timepoint is not None:
                if at_timepoint < len(statuses) and statuses[at_timepoint] == PoreStatus.ACTIVE:
                    active.append(pore_id)
            else:
                if statuses and statuses[-1] == PoreStatus.ACTIVE:
                    active.append(pore_id)
        return active
    
    def get_volume_ratio(self, pore_id: int, timepoint: int) -> float:
        """
        Get the volume ratio (current/reference) for a pore at a timepoint.
        Returns 0.0 if compressed, or the ratio of current volume to t=0 volume.
        """
        volumes = self.volume_history.get(pore_id, [])
        if not volumes or timepoint >= len(volumes):
            return 0.0
        if volumes[0] == 0:
            return 0.0
        return volumes[timepoint] / volumes[0]


@dataclass
class TimeSeriesPNM:
    """
    Complete 4D CT pore network model with temporal tracking.
    
    This is the top-level data structure that holds:
    - The reference (baseline) pore network from t=0
    - All subsequent snapshots
    - The tracking results linking pores across time
    
    Attributes:
        reference_snapshot: The baseline PNM from timepoint 0
        snapshots: List of all snapshots including reference
        tracking: Results of pore tracking across timepoints
        source_folders: List of DICOM folder paths for each timepoint
    """
    reference_snapshot: Optional[PNMSnapshot] = None
    snapshots: List[PNMSnapshot] = field(default_factory=list)
    tracking: PoreTrackingResult = field(default_factory=PoreTrackingResult)
    source_folders: List[str] = field(default_factory=list)
    
    @property
    def num_timepoints(self) -> int:
        """Number of timepoints in the series."""
        return len(self.snapshots)
    
    @property
    def num_reference_pores(self) -> int:
        """Number of pores in the reference snapshot."""
        if self.reference_snapshot:
            return self.reference_snapshot.num_pores
        return 0
    
    def get_snapshot(self, time_index: int) -> Optional[PNMSnapshot]:
        """Get snapshot at a specific time index."""
        if 0 <= time_index < len(self.snapshots):
            return self.snapshots[time_index]
        return None
    
    def get_pore_volume_at_time(self, pore_id: int, time_index: int) -> float:
        """Get volume of a specific pore at a specific timepoint."""
        volumes = self.tracking.volume_history.get(pore_id, [])
        if time_index < len(volumes):
            return volumes[time_index]
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the time series analysis."""
        if not self.reference_snapshot:
            return {"status": "No reference snapshot"}
        
        num_compressed = len(self.tracking.get_compressed_pore_ids())
        num_active = len(self.tracking.get_active_pore_ids())
        
        # Calculate average volume retention
        volume_ratios = []
        for pore_id in self.tracking.reference_ids:
            ratio = self.tracking.get_volume_ratio(pore_id, self.num_timepoints - 1)
            volume_ratios.append(ratio)
        
        avg_retention = np.mean(volume_ratios) if volume_ratios else 0.0
        
        return {
            "num_timepoints": self.num_timepoints,
            "reference_pores": self.num_reference_pores,
            "active_pores": num_active,
            "compressed_pores": num_compressed,
            "compression_rate": num_compressed / self.num_reference_pores if self.num_reference_pores > 0 else 0,
            "avg_volume_retention": avg_retention
        }
