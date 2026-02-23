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


def _annotations_from_labels(labels: np.ndarray, step_index: int) -> dict:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    voids = []
    for gt_id in ids:
        coords = np.argwhere(labels == gt_id)
        center_zyx = coords.mean(axis=0)
        voxels = float(len(coords))
        radius = float((3.0 * voxels / (4.0 * np.pi)) ** (1.0 / 3.0))
        voids.append(
            {
                "id": int(gt_id),
                "center_mm": [float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0])],
                "center_voxel": [float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0])],
                "radius_mm": radius,
                "volume_mm3": voxels,
            }
        )
    return {
        "step_index": int(step_index),
        "voxel_size": 1.0,
        "origin": [0.0, 0.0, 0.0],
        "volume_shape": [int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2])],
        "num_voids": int(len(voids)),
        "voids": voids,
    }


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

    volumes = [
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={
                "sim_annotations": {
                    "annotations": _annotations_from_labels(gt_t0, step_index=0),
                }
            },
        ),
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={
                "sim_annotations": {
                    "annotations": _annotations_from_labels(gt_t1, step_index=1),
                }
            },
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
    assert any("annotations.json missing" in msg for msg in report["warnings"])


def test_sim_annotation_primary_uses_annotations_not_labels(tmp_path):
    shape = (18, 18, 18)

    pred_t0 = np.zeros(shape, dtype=np.int32)
    pred_t0[2:6, 2:6, 2:6] = 1
    pred_t0[10:14, 10:14, 10:14] = 2

    pred_t1 = np.zeros(shape, dtype=np.int32)
    pred_t1[3:7, 2:6, 2:6] = 11
    pred_t1[10:14, 9:13, 10:14] = 22

    ann_t0_labels = np.zeros(shape, dtype=np.int32)
    ann_t0_labels[2:6, 2:6, 2:6] = 101
    ann_t0_labels[10:14, 10:14, 10:14] = 202
    ann_t1_labels = np.zeros(shape, dtype=np.int32)
    ann_t1_labels[3:7, 2:6, 2:6] = 101
    ann_t1_labels[10:14, 9:13, 10:14] = 202

    wrong_gt_t0 = np.zeros(shape, dtype=np.int32)
    wrong_gt_t0[2:6, 2:6, 2:6] = 900
    wrong_gt_t0[10:14, 10:14, 10:14] = 901
    wrong_gt_t1 = np.zeros(shape, dtype=np.int32)
    wrong_gt_t1[3:7, 2:6, 2:6] = 900
    wrong_gt_t1[10:14, 9:13, 10:14] = 901

    gt_path_t0 = tmp_path / "wrong_labels_t0.npy"
    gt_path_t1 = tmp_path / "wrong_labels_t1.npy"
    np.save(gt_path_t0, wrong_gt_t0)
    np.save(gt_path_t1, wrong_gt_t1)

    volumes = [
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={
                "sim_annotations": {
                    "annotations": _annotations_from_labels(ann_t0_labels, step_index=0),
                    "files": {"labels_npy": str(gt_path_t0)},
                }
            },
        ),
        VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            metadata={
                "sim_annotations": {
                    "annotations": _annotations_from_labels(ann_t1_labels, step_index=1),
                    "files": {"labels_npy": str(gt_path_t1)},
                }
            },
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
    primary_map_t0 = report["steps"][0]["mapping"]["pred_to_gt"]
    strict_map_t0 = report["steps"][0]["strict_mapping"]["pred_to_gt"]

    assert set(primary_map_t0.values()) == {101, 202}
    assert set(strict_map_t0.values()) == {900, 901}
