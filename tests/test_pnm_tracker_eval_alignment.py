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


def test_eval_uses_shape_alignment_for_pred_and_gt(tmp_path):
    pred_shape = (20, 18, 16)
    pred_t0 = np.zeros(pred_shape, dtype=np.int32)
    pred_t0[3:7, 4:8, 5:9] = 1

    pred_t1 = np.zeros(pred_shape, dtype=np.int32)
    pred_t1[4:8, 5:9, 6:10] = 11

    gt_t0 = np.where(np.transpose(pred_t0, (2, 1, 0)) > 0, 101, 0).astype(np.int16)
    gt_t1 = np.where(np.transpose(pred_t1, (2, 1, 0)) > 0, 101, 0).astype(np.int16)

    gt_path_t0 = tmp_path / "labels_t0.npy"
    gt_path_t1 = tmp_path / "labels_t1.npy"
    np.save(gt_path_t0, gt_t0)
    np.save(gt_path_t1, gt_t1)

    volumes = [
        VolumeData(
            raw_data=np.zeros(pred_shape, dtype=np.int16),
            metadata={"sim_annotations": {"files": {"labels_npy": str(gt_path_t0)}}},
        ),
        VolumeData(
            raw_data=np.zeros(pred_shape, dtype=np.int16),
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
    assert report["steps"][0]["evaluated"] is True
    assert report["steps"][1]["evaluated"] is True
    assert report["steps"][0]["alignment"]["gt_perm"] == [2, 1, 0]
    assert report["steps"][1]["tracking"]["accuracy"] == pytest.approx(1.0)
    assert not any("shape mismatch not alignable" in msg for msg in report["errors"])

