import numpy as np
import pytest

from core import VolumeData
from core.time_series import PNMSnapshot
from processors.pnm_tracker import PNMTracker


def _make_snapshot(regions: np.ndarray, time_index: int) -> PNMSnapshot:
    labels = np.unique(regions)
    labels = labels[labels > 0]
    centers = []
    radii = []
    volumes = []
    for label in labels:
        coords = np.argwhere(regions == label)
        centers.append(coords.mean(axis=0).astype(np.float64))
        volumes.append(float(len(coords)))
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


def test_pore_level_metrics_merge_fragmented_gt_ids(tmp_path):
    shape = (14, 14, 14)

    pred_t0 = np.zeros(shape, dtype=np.int32)
    pred_t0[2:6, 2:6, 2:6] = 1
    pred_t0[8:12, 8:12, 8:12] = 2

    pred_t1 = np.zeros(shape, dtype=np.int32)
    pred_t1[2:6, 3:7, 2:6] = 11
    pred_t1[8:12, 7:11, 8:12] = 22

    # Two fragmented GT ids per pore.
    gt_t0 = np.zeros(shape, dtype=np.int32)
    gt_t0[2:6, 2:4, 2:6] = 101
    gt_t0[2:6, 4:6, 2:6] = 102
    gt_t0[8:12, 8:10, 8:12] = 201
    gt_t0[8:12, 10:12, 8:12] = 202

    gt_t1 = np.zeros(shape, dtype=np.int32)
    gt_t1[2:6, 2:4, 2:6] = 101
    gt_t1[2:6, 4:6, 2:6] = 102
    gt_t1[8:12, 8:10, 8:12] = 201
    gt_t1[8:12, 10:12, 8:12] = 202

    gt_path_t0 = tmp_path / "labels_t0.npy"
    gt_path_t1 = tmp_path / "labels_t1.npy"
    np.save(gt_path_t0, gt_t0)
    np.save(gt_path_t1, gt_t1)

    volumes = [
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={"sim_annotations": {"files": {"labels_npy": str(gt_path_t0)}}},
        ),
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={"sim_annotations": {"files": {"labels_npy": str(gt_path_t1)}}},
        ),
    ]

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=0,
    )
    tracker.set_reference(_make_snapshot(pred_t0, time_index=0))
    tracker.track_snapshot(_make_snapshot(pred_t1, time_index=1))

    report = tracker.evaluate_against_sim_annotations(volumes, instance_iou_threshold=0.1)
    assert report["available"] is True

    # Primary tracking now follows pore-level merged GT policy.
    assert report["steps"][1]["tracking"]["accuracy"] == pytest.approx(1.0)
    # Strict metric is still exposed for diagnostics.
    assert report["steps"][1]["strict_tracking"]["accuracy"] == pytest.approx(0.5)
    assert report["steps"][1]["pore_level"]["tracking"]["accuracy"] == pytest.approx(1.0)

    overall = report["overall"]
    assert overall["mean_tracking_accuracy"] == pytest.approx(1.0)
    assert overall["mean_tracking_accuracy_strict"] == pytest.approx(0.5)
    assert overall["mean_pore_level_tracking_accuracy"] == pytest.approx(1.0)
    assert "mean_pore_level_instance_f1" in overall
    assert "t0_pore_level_reference_gt_coverage" in overall
