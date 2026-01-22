"""
Pore Network Model Tracker for 4D CT analysis.

Tracks pore changes across time series by matching pores between
the reference (t=0) and subsequent timepoints using spatial overlap.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
import time

from core.time_series import (
    PNMSnapshot, PoreTrackingResult, TimeSeriesPNM, PoreStatus
)
from config import (
    TRACKING_IOU_THRESHOLD,
    TRACKING_COMPRESSION_THRESHOLD
)


class PNMTracker:
    """
    Tracks pore network changes across 4D CT time series.
    
    This tracker maintains correspondence between pores in a reference
    snapshot (t=0) and all subsequent timepoints. It handles:
    - Pore shrinkage (volume decrease)
    - Full compression (volume → 0)
    - Volume recovery (if decompression occurs)
    
    Note: This tracker assumes pores do not appear or disappear,
    only change size. This is appropriate for sponge compression.
    """
    
    def __init__(self, 
                 iou_threshold: float = None,
                 compression_threshold: float = None):
        """
        Initialize the tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider pores matched (default from config)
            compression_threshold: Volume ratio below which pore is "compressed"
        """
        self.iou_threshold = iou_threshold or TRACKING_IOU_THRESHOLD
        self.compression_threshold = compression_threshold or TRACKING_COMPRESSION_THRESHOLD
        
        self.time_series: TimeSeriesPNM = TimeSeriesPNM()
        self._reference_masks: Dict[int, np.ndarray] = {}  # Cache of reference pore masks
    
    def set_reference(self, snapshot: PNMSnapshot) -> None:
        """
        Set the reference (baseline) snapshot at t=0.
        
        This establishes the ground truth for pore tracking.
        All subsequent timepoints will be matched against this reference.
        
        Args:
            snapshot: PNMSnapshot from timepoint 0
        """
        snapshot.time_index = 0
        self.time_series.reference_snapshot = snapshot
        self.time_series.snapshots = [snapshot]
        
        # Initialize tracking result
        self.time_series.tracking = PoreTrackingResult(
            reference_ids=snapshot.pore_ids.tolist()
        )
        
        # Initialize history with reference values
        for pore_id, volume in zip(snapshot.pore_ids, snapshot.pore_volumes):
            pore_id = int(pore_id)
            self.time_series.tracking.volume_history[pore_id] = [float(volume)]
            self.time_series.tracking.status_history[pore_id] = [PoreStatus.ACTIVE]
            self.time_series.tracking.iou_history[pore_id] = [1.0]  # Perfect match with self
        
        # Cache reference pore masks if segmentation is available
        if snapshot.segmented_regions is not None:
            self._cache_reference_masks(snapshot)
        
        print(f"[Tracker] Reference set: {snapshot.num_pores} pores")
    
    def _cache_reference_masks(self, snapshot: PNMSnapshot) -> None:
        """Cache binary masks for each reference pore for efficient matching."""
        self._reference_masks.clear()
        regions = snapshot.segmented_regions
        
        for pore_id in snapshot.pore_ids:
            # Store bounding box + mask for memory efficiency
            mask = regions == pore_id
            # Find bounding box
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
            
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            
            self._reference_masks[int(pore_id)] = {
                'bbox': (mins, maxs),
                'mask': mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            }
    
    def track_snapshot(self, 
                       snapshot: PNMSnapshot,
                       callback: Optional[Callable[[int, str], None]] = None) -> None:
        """
        Track pores from reference to a new timepoint.
        
        This matches each reference pore to the current segmentation
        and updates the tracking history.
        
        Args:
            snapshot: PNMSnapshot from the new timepoint
            callback: Optional progress callback (percent, message)
        """
        if self.time_series.reference_snapshot is None:
            raise ValueError("Reference snapshot not set. Call set_reference first.")
        
        if snapshot.segmented_regions is None:
            raise ValueError("Snapshot must include segmented_regions for tracking.")
        
        start_time = time.time()
        time_index = len(self.time_series.snapshots)
        snapshot.time_index = time_index
        
        ref_snapshot = self.time_series.reference_snapshot
        current_regions = snapshot.segmented_regions
        
        # Initialize mapping for this timepoint
        id_map: Dict[int, int] = {}
        
        total_pores = len(ref_snapshot.pore_ids)
        
        for i, ref_id in enumerate(ref_snapshot.pore_ids):
            ref_id = int(ref_id)
            
            if callback and i % 50 == 0:
                progress = int(100 * i / total_pores)
                callback(progress, f"Matching pore {i+1}/{total_pores}...")
            
            # Get reference pore region
            if ref_id in self._reference_masks:
                ref_data = self._reference_masks[ref_id]
                bbox = ref_data['bbox']
                ref_mask_local = ref_data['mask']
                
                # Extract corresponding region from current
                mins, maxs = bbox
                current_region = current_regions[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
                
                # Expand reference mask to full size for proper comparison
                ref_mask_full = np.zeros_like(current_region, dtype=bool)
                ref_mask_full[:ref_mask_local.shape[0], :ref_mask_local.shape[1], :ref_mask_local.shape[2]] = ref_mask_local
                
                matched_id, iou, current_volume = self._match_pore(
                    ref_mask_full, current_region
                )
            else:
                # Fallback: use full mask comparison (slower)
                ref_mask = ref_snapshot.segmented_regions == ref_id
                matched_id, iou, current_volume = self._match_pore(
                    ref_mask, current_regions
                )
            
            # Determine status
            if matched_id is None or iou < self.iou_threshold:
                # No match found - fully compressed
                status = PoreStatus.COMPRESSED
                current_volume = 0.0
                matched_id = -1  # Indicate no match
            else:
                # Check if volume is below compression threshold
                ref_volume = float(ref_snapshot.pore_volumes[i])
                if ref_volume > 0 and (current_volume / ref_volume) < self.compression_threshold:
                    status = PoreStatus.COMPRESSED
                    current_volume = 0.0
                else:
                    status = PoreStatus.ACTIVE
            
            # Update tracking
            self.time_series.tracking.volume_history[ref_id].append(float(current_volume))
            self.time_series.tracking.status_history[ref_id].append(status)
            self.time_series.tracking.iou_history[ref_id].append(float(iou))
            id_map[ref_id] = matched_id
        
        # Store mapping and snapshot
        self.time_series.tracking.id_mapping[time_index] = id_map
        self.time_series.snapshots.append(snapshot)
        
        # Statistics
        num_compressed = sum(
            1 for status in self.time_series.tracking.status_history.values()
            if status[-1] == PoreStatus.COMPRESSED
        )
        num_active = total_pores - num_compressed
        
        elapsed = time.time() - start_time
        print(f"[Tracker] t={time_index}: {num_active} active, {num_compressed} compressed ({elapsed:.2f}s)")
        
        if callback:
            callback(100, f"Tracked {total_pores} pores")
    
    def _match_pore(self, 
                    ref_mask: np.ndarray, 
                    current_labels: np.ndarray) -> Tuple[Optional[int], float, float]:
        """
        Match a reference pore to current segmentation labels.
        
        Uses Intersection over Union (IoU) to find the best matching
        pore in the current timepoint.
        
        Args:
            ref_mask: Binary mask of the reference pore region
            current_labels: Current timepoint's watershed label array
            
        Returns:
            (matched_label_id, iou_score, current_volume_voxels)
            Returns (None, 0.0, 0.0) if no overlap found
        """
        # Find all labels that overlap with reference mask
        overlapping_labels = current_labels[ref_mask]
        overlapping_labels = overlapping_labels[overlapping_labels > 0]
        
        if len(overlapping_labels) == 0:
            return None, 0.0, 0.0
        
        # Find the dominant (most common) overlapping label
        label_counts = np.bincount(overlapping_labels)
        best_label = int(np.argmax(label_counts[1:]) + 1) if len(label_counts) > 1 else int(overlapping_labels[0])
        
        # Calculate IoU
        current_mask = current_labels == best_label
        
        intersection = np.sum(ref_mask & current_mask)
        union = np.sum(ref_mask | current_mask)
        
        if union == 0:
            return None, 0.0, 0.0
        
        iou = intersection / union
        current_volume = float(np.sum(current_mask))
        
        return best_label, iou, current_volume
    
    def get_results(self) -> TimeSeriesPNM:
        """Get the complete time series analysis results."""
        return self.time_series
    
    def get_volume_history(self, pore_id: int) -> List[float]:
        """Get volume history for a specific pore across all timepoints."""
        return self.time_series.tracking.get_volume_series(pore_id)
    
    def get_pore_status(self, pore_id: int, timepoint: int) -> PoreStatus:
        """Get the status of a pore at a specific timepoint."""
        statuses = self.time_series.tracking.status_history.get(pore_id, [])
        if timepoint < len(statuses):
            return statuses[timepoint]
        return PoreStatus.UNKNOWN
    
    def export_volume_csv(self, filepath: str) -> None:
        """
        Export volume history to CSV file.
        
        Columns: pore_id, t0, t1, t2, ..., final_status
        """
        import csv
        
        tracking = self.time_series.tracking
        num_timepoints = self.time_series.num_timepoints
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['pore_id'] + [f't{i}' for i in range(num_timepoints)] + ['final_status', 'volume_retention']
            writer.writerow(header)
            
            # Data rows
            for pore_id in tracking.reference_ids:
                volumes = tracking.volume_history.get(pore_id, [])
                statuses = tracking.status_history.get(pore_id, [])
                
                final_status = statuses[-1].value if statuses else 'unknown'
                
                # Volume retention ratio
                if volumes and volumes[0] > 0:
                    retention = volumes[-1] / volumes[0] if volumes else 0
                else:
                    retention = 0
                
                row = [pore_id] + volumes + [final_status, f'{retention:.4f}']
                writer.writerow(row)
        
        print(f"[Tracker] Exported volume history to {filepath}")
