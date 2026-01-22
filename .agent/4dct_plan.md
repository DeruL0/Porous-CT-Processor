# 4D CT Pore Network Tracking - Implementation Plan

## Overview

Add 4D CT functionality to track pore network model (PNM) changes across time-series DICOM sequences. This simulates sponge compression where:

- Pores can **shrink** (volume decreases)
- Pores can be **fully compressed** (volume → 0)
- Pores **cannot appear or disappear** (only tracking, no generation/deletion)

## Architecture Design

### Data Model

```
TimeSeriesPNM
├── reference_pnm: PNMSnapshot (t=0, baseline)
├── snapshots: List[PNMSnapshot] (t=1, t=2, ...)
└── tracking: PoreTrackingResult
    ├── pore_id_map: {snapshot_idx: {reference_id → current_id}}
    ├── volume_history: {pore_id: [v0, v1, v2, ...]}
    └── status_history: {pore_id: [ACTIVE, ACTIVE, COMPRESSED, ...]}

PNMSnapshot
├── timestamp: int (time index)
├── pore_centers: np.ndarray (N, 3)
├── pore_radii: np.ndarray (N,)
├── pore_ids: np.ndarray (N,)
├── connections: List[Tuple[int, int]]
└── segmented_regions: np.ndarray (3D labels)
```

### Core Tracking Algorithm

1. **Reference Establishment** (t=0)
   - Load first timepoint → generate PNM → store as reference

2. **Temporal Matching** (t=n)
   - For each pore in reference:
     - Find overlapping region in current segmentation
     - Calculate IoU (Intersection over Union)
     - If IoU > threshold → matched (same pore, possibly resized)
     - If no overlap or IoU ≈ 0 → mark as COMPRESSED

3. **Volume Update**
   - For matched pores: update volume from current segmentation
   - For compressed pores: set volume = 0 (still tracked)

## New Files

### [CREATE] `core/time_series.py`

- `TimeSeriesPNM` dataclass
- `PNMSnapshot` dataclass
- `PoreStatus` enum (ACTIVE, COMPRESSED)

### [CREATE] `processors/pnm_tracker.py`

- `PNMTracker` class
  - `set_reference(pnm_snapshot)` - Set t=0 baseline
  - `track_snapshot(current_snapshot)` - Match and track pores
  - `get_volume_history(pore_id)` - Get volume over time
  - `get_compressed_pores()` - List of fully compressed pores

### [CREATE] `loaders/dicom_series_loader.py`

- `TimeSeriesDicomLoader` class
  - Load multiple DICOM folders as time sequence
  - Return `List[VolumeData]` ordered by time

### [CREATE] `gui/panels/timeseries_panel.py`

- Timeline slider for timepoint selection
- Play/pause animation
- Volume chart for selected pore
- Compression status indicators

## Modified Files

### [MODIFY] `processors/pnm.py`

- Extract `_extract_pore_data` results into `PNMSnapshot`
- Add method to return snapshot instead of mesh

### [MODIFY] `App.py`

- Add "Load 4D CT Series" menu action
- Add timeseries panel to workflow dock
- Connect timeline to render updates

### [MODIFY] `rendering/render_engine.py`

- Support rendering PNM with color-coded compression status
- Animate between timepoints

## Workflow

```
User Action                     System Response
─────────────────────────────────────────────────────
1. Load 4D CT Series    →  Select folder containing t0, t1, t2... subfolders
                        →  Load all timepoints as List[VolumeData]
                        
2. Generate Reference   →  Run PNM on t=0
PNM                     →  Store as TimeSeriesPNM.reference_pnm
                        
3. Track All Timepoints →  For each t=1..N:
                           - Run PNM on t=n
                           - Match pores to reference
                           - Update volume_history
                           
4. Visualize            →  Show reference PNM mesh
                        →  Color code: size = current volume, color = status
                        →  Timeline slider for animation
```

## Matching Algorithm Detail

```python
def match_pore(ref_pore_mask, current_labels):
    """
    Match a reference pore to current segmentation.
    
    Args:
        ref_pore_mask: Binary mask of reference pore region
        current_labels: Current timepoint's watershed labels
        
    Returns:
        (matched_id, iou, current_volume) or (None, 0, 0) if compressed
    """
    # Find overlapping labels
    overlapping_labels = current_labels[ref_pore_mask]
    overlapping_labels = overlapping_labels[overlapping_labels > 0]
    
    if len(overlapping_labels) == 0:
        return None, 0.0, 0  # Fully compressed
    
    # Find dominant label
    label_counts = np.bincount(overlapping_labels)
    best_label = np.argmax(label_counts[1:]) + 1
    
    # Calculate IoU
    ref_mask = ref_pore_mask
    cur_mask = current_labels == best_label
    
    intersection = np.sum(ref_mask & cur_mask)
    union = np.sum(ref_mask | cur_mask)
    iou = intersection / union if union > 0 else 0
    
    # Current volume
    current_volume = np.sum(cur_mask)
    
    return best_label, iou, current_volume
```

## Configuration

Add to `config.py`:

```python
# 4D CT Tracking
TRACKING_IOU_THRESHOLD = 0.1      # Min IoU to consider pores matched
TRACKING_COMPRESSION_THRESHOLD = 0.01  # Volume ratio below which pore is "compressed"
```

## Priority Order

1. **P1**: Core data structures (`core/time_series.py`)
2. **P1**: Tracking algorithm (`processors/pnm_tracker.py`)
3. **P2**: Time series loader (`loaders/dicom_series_loader.py`)
4. **P2**: PNM snapshot extraction (modify `pnm.py`)
5. **P3**: GUI panel (`gui/panels/timeseries_panel.py`)
6. **P3**: Visualization updates (`render_engine.py`)
7. **P3**: App integration (`App.py`)
