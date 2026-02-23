import numpy as np

from core.time_series import PNMSnapshot
from processors.pnm_tracker import PNMTracker


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


def test_fixed_reference_semantics_keep_reference_ids_and_emit_novel_diagnostics():
    ref_regions = np.zeros((16, 16, 16), dtype=np.int32)
    ref_regions[2:5, 2:5, 2:5] = 1
    ref_regions[10:13, 10:13, 10:13] = 2
    ref_snap = _snapshot_from_regions(ref_regions, time_index=0)

    cur_regions = np.zeros((16, 16, 16), dtype=np.int32)
    cur_regions[2:5, 2:5, 2:5] = 11
    cur_regions[10:13, 10:13, 10:13] = 22
    cur_regions[6:9, 6:9, 6:9] = 33  # novel segment
    cur_snap = _snapshot_from_regions(cur_regions, time_index=1)

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        gating_params={
            "novel_min_volume_voxels": 1.0,
            "novel_min_persistence": 1,
        },
    )
    tracker.set_reference(ref_snap)
    tracker.track_snapshot(cur_snap)

    results = tracker.get_results()
    id_map = results.tracking.id_mapping[1]
    assert set(results.tracking.reference_ids) == {1, 2}
    assert set(id_map.keys()) == {1, 2}

    novel = results.snapshots[1].metadata.get("novel_segmentation", {})
    assert novel.get("reference_policy") == "fixed_reference_set"
    assert int(novel.get("num_unmatched_current_segments", 0)) >= 1
    assert int(novel.get("num_untracked_novel_segments", 0)) >= 1
