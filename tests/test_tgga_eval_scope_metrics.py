from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.time_series import PNMSnapshot
from processors.pnm_tracker import PNMTracker


@dataclass
class _DummyVolume:
    labels_path: Path

    @property
    def metadata(self):
        return {"sim_annotations": {"files": {"labels_npy": str(self.labels_path)}}}


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


def test_eval_reports_reference_scope_and_t0_coverage(tmp_path):
    shape = (12, 12, 12)

    pred_t0 = np.zeros(shape, dtype=np.int32)
    pred_t0[1:4, 1:4, 1:4] = 1
    pred_t0[7:10, 7:10, 7:10] = 2

    pred_t1 = np.zeros(shape, dtype=np.int32)
    pred_t1[1:4, 1:4, 1:4] = 11
    pred_t1[7:10, 7:10, 7:10] = 22
    pred_t1[4:6, 4:6, 4:6] = 33  # novel segment (diagnostic only)

    gt_t0 = np.zeros(shape, dtype=np.int32)
    gt_t0[1:4, 1:4, 1:4] = 101
    gt_t0[7:10, 7:10, 7:10] = 202
    gt_t0[0, 0, 0] = 303  # unmatched GT -> t0 reference coverage = 2 / 3

    gt_t1 = np.zeros(shape, dtype=np.int32)
    gt_t1[1:4, 1:4, 1:4] = 101
    gt_t1[7:10, 7:10, 7:10] = 202
    gt_t1[0, 0, 0] = 303

    gt_path_t0 = tmp_path / "labels_t0.npy"
    gt_path_t1 = tmp_path / "labels_t1.npy"
    np.save(gt_path_t0, gt_t0)
    np.save(gt_path_t1, gt_t1)

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        gating_params={"novel_min_volume_voxels": 1.0, "novel_min_persistence": 1},
    )
    tracker.set_reference(_snapshot_from_regions(pred_t0, time_index=0))
    tracker.track_snapshot(_snapshot_from_regions(pred_t1, time_index=1))

    report = tracker.evaluate_against_sim_annotations(
        volumes=[_DummyVolume(gt_path_t0), _DummyVolume(gt_path_t1)],
        instance_iou_threshold=0.1,
    )

    assert report["available"] is True
    overall = report["overall"]
    assert overall["t0_reference_gt_coverage"] == pytest.approx(2.0 / 3.0)
    assert "mean_reference_scope_gt_coverage" in overall

    step1_tracking = report["steps"][1]["tracking"]
    assert step1_tracking["reference_scope_total_gt_ids"] == 3
    assert "reference_scope_gt_coverage" in step1_tracking
    assert "untracked_novel_segments" in step1_tracking
