import numpy as np
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
        volumes.append(float(len(coords)))
        centers.append(coords.mean(axis=0).astype(np.float64))
        r_vox = (3.0 * len(coords) / (4.0 * np.pi)) ** (1.0 / 3.0)
        radii.append(float(r_vox))

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


def test_sim_annotation_evaluation_reports_tracking_accuracy(tmp_path):
    shape = (20, 20, 20)

    pred_t0 = np.zeros(shape, dtype=np.int32)
    pred_t0[2:6, 2:6, 2:6] = 1
    pred_t0[10:14, 10:14, 10:14] = 2

    pred_t1 = np.zeros(shape, dtype=np.int32)
    pred_t1[2:6, 2:6, 2:6] = 11
    pred_t1[10:14, 10:14, 10:14] = 22

    gt_t0 = np.zeros(shape, dtype=np.int16)
    gt_t0[2:6, 2:6, 2:6] = 101
    gt_t0[10:14, 10:14, 10:14] = 202

    gt_t1 = np.zeros(shape, dtype=np.int16)
    gt_t1[2:6, 2:6, 2:6] = 101
    gt_t1[10:14, 10:14, 10:14] = 202

    gt_path_t0 = tmp_path / "Step_00_labels.npy"
    gt_path_t1 = tmp_path / "Step_01_labels.npy"
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
    assert report["status"] == "ok"
    assert report["overall"]["num_steps_evaluated"] == 2
    assert report["steps"][0]["instance"]["f1"] == pytest.approx(1.0)
    assert report["steps"][1]["tracking"]["accuracy"] == pytest.approx(1.0)
    per_ref = report["steps"][1]["tracking"]["per_reference"]
    assert per_ref[1]["outcome"] == "correct_active"
    assert per_ref[2]["outcome"] == "correct_active"
    assert per_ref[1]["correct"] is True
    assert tracker.get_results().tracking.evaluation["available"] is True


def test_sim_annotation_evaluation_handles_missing_label_path_gracefully():
    shape = (12, 12, 12)
    pred_t0 = np.zeros(shape, dtype=np.int32)
    pred_t0[2:5, 2:5, 2:5] = 1

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
    )
    tracker.set_reference(_make_snapshot(pred_t0, time_index=0))

    volumes = [VolumeData(raw_data=np.zeros(shape, dtype=np.int16), metadata={})]
    report = tracker.evaluate_against_sim_annotations(volumes, instance_iou_threshold=0.1)

    assert report["available"] is False
    assert report["status"] == "unavailable"
    assert any("labels.npy path missing" in msg for msg in report["warnings"])
