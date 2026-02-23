import numpy as np

from core.time_series import PNMSnapshot
from processors.pnm_tracker import PNMTracker, build_candidates


def _snapshot_from_regions(regions: np.ndarray, time_index: int) -> PNMSnapshot:
    labels = np.unique(regions)
    labels = labels[labels > 0]
    centers = []
    radii = []
    volumes = []
    for label in labels:
        coords = np.argwhere(regions == label)
        volumes.append(float(len(coords)))
        centers.append(coords.mean(axis=0).astype(np.float64))
        radii.append(1.0)
    return PNMSnapshot(
        time_index=time_index,
        pore_centers=np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 3), dtype=np.float64),
        pore_radii=np.asarray(radii, dtype=np.float64) if radii else np.zeros((0,), dtype=np.float64),
        pore_ids=np.asarray(labels, dtype=np.int32),
        pore_volumes=np.asarray(volumes, dtype=np.float64) if volumes else np.zeros((0,), dtype=np.float64),
        connections=[],
        segmented_regions=regions,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        metadata={},
    )


def test_small_pore_soft_gate_allows_low_iou_candidate_with_intersection():
    ref_regions = np.zeros((8, 8, 8), dtype=np.int32)
    ref_regions[2, 2, 2] = 1
    # Keep reference sparse in a wider bbox to make local IoU meaningfully low.
    ref_regions[2, 2, 6] = 1  # ref volume = 2 voxels
    ref_snap = _snapshot_from_regions(ref_regions, time_index=0)

    cur_regions = np.zeros((8, 8, 8), dtype=np.int32)
    # candidate volume = 4, overlap = 1 => IoU = 1 / (2 + 4 - 1) = 0.2
    cur_regions[2, 2, 2] = 9
    cur_regions[2, 2, 3] = 9
    cur_regions[2, 2, 4] = 9
    cur_regions[2, 2, 5] = 9
    cur_regions[2, 2, 6] = 0  # keep only one-voxel overlap with sparse reference
    cur_snap = _snapshot_from_regions(cur_regions, time_index=1)

    tracker = PNMTracker(match_mode="temporal_global", assign_solver="scipy", use_gpu=False, use_batch=False)
    tracker.set_reference(ref_snap)

    data = build_candidates(
        reference_snapshot=ref_snap,
        current_snapshot=cur_snap,
        reference_masks=tracker._reference_masks,
        current_regions=cur_snap.segmented_regions,
        spacing_xyz=(1.0, 1.0, 1.0),
        # Keep zero shift so overlap stays intentionally low.
        predicted_centers={1: ref_snap.pore_centers[0]},
        cost_weights=(0.45, 0.30, 0.20, 0.05),
        gate_center_radius_factor=3.0,
        gate_volume_ratio_min=1e-4,
        gate_volume_ratio_max=10.0,
        gate_iou_min=0.3,
        small_pore_volume_voxels=10.0,
        soft_gate_min_intersection_voxels=1,
        soft_gate_cost_penalty=0.1,
    )

    assert bool(data["row_geom_candidate"][0]) is True
    assert bool(data["row_valid_candidate"][0]) is True
    metrics = data["pair_metrics"][(0, 0)]
    assert metrics["soft_gate_used"] is True
    assert int(metrics["intersection_voxels"]) == 1


def test_small_pore_soft_gate_rejects_when_intersection_below_threshold():
    ref_regions = np.zeros((8, 8, 8), dtype=np.int32)
    ref_regions[2, 2, 2] = 1
    ref_regions[2, 2, 6] = 1
    ref_snap = _snapshot_from_regions(ref_regions, time_index=0)

    cur_regions = np.zeros((8, 8, 8), dtype=np.int32)
    cur_regions[2, 2, 2] = 9
    cur_regions[2, 2, 3] = 9
    cur_regions[2, 2, 4] = 9
    cur_regions[2, 2, 5] = 9
    cur_regions[2, 2, 6] = 0
    cur_snap = _snapshot_from_regions(cur_regions, time_index=1)

    tracker = PNMTracker(match_mode="temporal_global", assign_solver="scipy", use_gpu=False, use_batch=False)
    tracker.set_reference(ref_snap)

    data = build_candidates(
        reference_snapshot=ref_snap,
        current_snapshot=cur_snap,
        reference_masks=tracker._reference_masks,
        current_regions=cur_snap.segmented_regions,
        spacing_xyz=(1.0, 1.0, 1.0),
        predicted_centers={1: ref_snap.pore_centers[0]},
        cost_weights=(0.45, 0.30, 0.20, 0.05),
        gate_center_radius_factor=3.0,
        gate_volume_ratio_min=1e-4,
        gate_volume_ratio_max=10.0,
        gate_iou_min=0.3,
        small_pore_volume_voxels=10.0,
        soft_gate_min_intersection_voxels=2,  # stricter than overlap=1
        soft_gate_cost_penalty=0.1,
    )

    assert bool(data["row_geom_candidate"][0]) is True
    assert bool(data["row_valid_candidate"][0]) is False
