import numpy as np
import pytest

from core.time_series import PNMSnapshot, PoreStatus
from processors.pnm_tracker import (
    HAS_LAPJV,
    HAS_SCIPY,
    PNMTracker,
    build_candidates,
    solve_global_assignment,
)
from processors.tracking_utils import estimate_macro_registration


def _make_snapshot(regions: np.ndarray, time_index: int = 0) -> PNMSnapshot:
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

    if centers:
        centers_arr = np.asarray(centers, dtype=np.float64)
    else:
        centers_arr = np.zeros((0, 3), dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64) if radii else np.zeros((0,), dtype=np.float64)
    volumes_arr = np.asarray(volumes, dtype=np.float64) if volumes else np.zeros((0,), dtype=np.float64)

    return PNMSnapshot(
        time_index=time_index,
        pore_centers=centers_arr,
        pore_radii=radii_arr,
        pore_ids=np.asarray(labels, dtype=np.int32),
        pore_volumes=volumes_arr,
        connections=[],
        segmented_regions=regions,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        metadata={},
    )


def _make_reference_and_current():
    ref = np.zeros((24, 24, 24), dtype=np.int32)
    ref[2:6, 2:6, 2:6] = 1
    ref[14:18, 14:18, 14:18] = 2

    cur = np.zeros((24, 24, 24), dtype=np.int32)
    cur[3:7, 3:7, 3:7] = 11
    cur[13:17, 13:17, 13:17] = 22
    return _make_snapshot(ref, 0), _make_snapshot(cur, 1)


def _paint_cube(volume: np.ndarray, label: int, center_zyx: tuple[int, int, int], size: int) -> None:
    half = size // 2
    z, y, x = center_zyx
    volume[z - half:z - half + size, y - half:y - half + size, x - half:x - half + size] = label


def test_tgga_one_to_one_mapping():
    ref_snap, cur_snap = _make_reference_and_current()
    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=0,
    )
    tracker.set_reference(ref_snap)
    tracker.track_snapshot(cur_snap)

    mapping = tracker.get_results().tracking.id_mapping[1]
    matched_current = [cid for cid in mapping.values() if cid != -1]
    assert len(matched_current) == len(set(matched_current))


def test_candidate_gating_rejects_far_pairs():
    ref = np.zeros((20, 20, 20), dtype=np.int32)
    ref[2:6, 2:6, 2:6] = 1
    ref_snap = _make_snapshot(ref, 0)

    cur = np.zeros((20, 20, 20), dtype=np.int32)
    cur[14:18, 14:18, 14:18] = 9
    cur_snap = _make_snapshot(cur, 1)

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
        gate_center_radius_factor=2.5,
        gate_volume_ratio_min=0.2,
        gate_volume_ratio_max=5.0,
        gate_iou_min=0.02,
    )
    assert data["row_geom_candidate"].shape[0] == 1
    assert not bool(data["row_geom_candidate"][0])
    assert np.all(data["cost_matrix"] >= 1e6)


@pytest.mark.skipif(not (HAS_LAPJV and HAS_SCIPY), reason="lapjv and scipy both required")
def test_solver_consistency_lapjv_vs_scipy():
    rng = np.random.default_rng(7)
    cost = rng.uniform(0.0, 1.0, size=(7, 9))

    m_lap, _ = solve_global_assignment(cost.copy(), assign_solver="lapjv")
    m_sci, _ = solve_global_assignment(cost.copy(), assign_solver="scipy")

    c_lap = sum(v for _, _, v in m_lap)
    c_sci = sum(v for _, _, v in m_sci)
    assert abs(c_lap - c_sci) < 1e-6


def test_legacy_greedy_basic_compatibility():
    ref_snap, cur_snap = _make_reference_and_current()
    tracker = PNMTracker(
        match_mode="legacy_greedy",
        use_gpu=False,
        use_batch=False,
    )
    tracker.set_reference(ref_snap)
    tracker.track_snapshot(cur_snap)

    tracking = tracker.get_results().tracking
    mapping = tracking.id_mapping[1]
    assert set(mapping.keys()) == {1, 2}
    assert tracking.status_history[1][-1] in {PoreStatus.ACTIVE, PoreStatus.COMPRESSED}
    assert tracking.status_history[2][-1] in {PoreStatus.ACTIVE, PoreStatus.COMPRESSED}


def test_extreme_compression_keeps_finite_cost_and_valid_assignment():
    ref = np.zeros((64, 64, 64), dtype=np.int32)
    ref[8:18, 8:18, 8:18] = 1  # 1000 voxels
    cur = np.zeros((64, 64, 64), dtype=np.int32)
    cur[30, 30, 30] = 9  # 1 voxel => ratio 0.001

    ref_snap = _make_snapshot(ref, 0)
    cur_snap = _make_snapshot(cur, 1)

    tracker = PNMTracker(match_mode="temporal_global", assign_solver="scipy", use_gpu=False, use_batch=False)
    tracker.set_reference(ref_snap)

    data = build_candidates(
        reference_snapshot=ref_snap,
        current_snapshot=cur_snap,
        reference_masks=tracker._reference_masks,
        current_regions=cur_snap.segmented_regions,
        spacing_xyz=(1.0, 1.0, 1.0),
        predicted_centers={1: cur_snap.pore_centers[0]},
        cost_weights=(0.45, 0.30, 0.20, 0.05),
        gate_center_radius_factor=8.0,
        gate_volume_ratio_min=0.2,
        gate_volume_ratio_max=5.0,
        gate_iou_min=0.02,
    )
    metrics = data["pair_metrics"][(0, 0)]
    assert np.isfinite(metrics["cost"])
    assert metrics["cost"] < 1e6
    assert 0.0005 <= metrics["volume_ratio"] <= 0.0015

    tracker.track_snapshot(cur_snap)
    mapping = tracker.get_results().tracking.id_mapping[1]
    assert mapping[1] == 9


def test_macro_registration_recovers_large_translation():
    ref = np.zeros((128, 128, 128), dtype=np.int32)
    _paint_cube(ref, 1, (24, 24, 24), size=8)
    cur = np.zeros((128, 128, 128), dtype=np.int32)
    _paint_cube(cur, 7, (74, 74, 74), size=8)  # +50 voxels translation

    ref_snap = _make_snapshot(ref, 0)
    cur_snap = _make_snapshot(cur, 1)

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=0,
    )
    tracker.set_reference(ref_snap)
    tracker.track_snapshot(cur_snap)

    mapping = tracker.get_results().tracking.id_mapping[1]
    assert mapping[1] == 7


def test_macro_registration_gpu_flag_falls_back_cleanly():
    ref = np.zeros((48, 48, 48), dtype=np.int32)
    _paint_cube(ref, 1, (12, 12, 12), size=6)
    cur = np.zeros((48, 48, 48), dtype=np.int32)
    _paint_cube(cur, 2, (20, 20, 20), size=6)

    result = estimate_macro_registration(
        reference_regions=ref,
        current_regions=cur,
        smoothing_sigma=1.5,
        upsample_factor=2,
        use_gpu=True,
        gpu_min_size_mb=0.0,
    )
    assert result.displacement.shape == (3,)
    assert np.linalg.norm(result.displacement) > 0.0


def test_kalman_predicts_nonlinear_motion_through_two_occluded_frames():
    shape = (80, 80, 80)

    t0 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t0, 1, (30, 30, 20), size=5)
    t1 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t1, 11, (30, 30, 23), size=5)
    t2 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t2, 12, (30, 30, 27), size=5)
    t3 = np.zeros(shape, dtype=np.int32)  # occlusion
    t4 = np.zeros(shape, dtype=np.int32)  # occlusion

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=5,
    )
    tracker.set_reference(_make_snapshot(t0, 0))
    tracker.track_snapshot(_make_snapshot(t1, 1))
    tracker.track_snapshot(_make_snapshot(t2, 2))
    tracker.track_snapshot(_make_snapshot(t3, 3))
    tracker.track_snapshot(_make_snapshot(t4, 4))

    tracking = tracker.get_results().tracking
    history = np.asarray(tracking.center_history[1], dtype=np.float64)
    pred_t3_x = float(history[3, 2])
    pred_t4_x = float(history[4, 2])

    # Quadratic-like target path: x = [20, 23, 27, 32, 38]
    assert abs(pred_t3_x - 32.0) <= 4.0
    assert abs(pred_t4_x - 38.0) <= 6.0
    assert tracking.id_mapping[3][1] == -1
    assert tracking.id_mapping[4][1] == -1


def test_pore_closure_event_is_labeled_closed_by_compression():
    shape = (48, 48, 48)

    t0 = np.zeros(shape, dtype=np.int32)
    t0[10:12, 10:12, 10:12] = 1  # tiny pore, volume=8
    t0[30:36, 30:36, 30:36] = 2  # stable large pore

    t1 = np.zeros(shape, dtype=np.int32)
    t1[11, 11, 11] = 11  # tiny pore shrinks near segmentation limit
    t1[30:36, 30:36, 30:36] = 22

    t2 = np.zeros(shape, dtype=np.int32)
    t2[30:36, 30:36, 30:36] = 33  # tiny pore disappears

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=3,
    )
    tracker.set_reference(_make_snapshot(t0, 0))
    tracker.track_snapshot(_make_snapshot(t1, 1))
    tracker.track_snapshot(_make_snapshot(t2, 2))

    tracking = tracker.get_results().tracking
    assert tracking.unmatched_reason[1][-1] == "closed_by_compression"
    assert tracking.status_history[1][-1] == PoreStatus.COMPRESSED
    assert tracking.volume_history[1][-1] == 0.0


def test_kalman_brakes_after_consecutive_tracking_loss():
    shape = (80, 80, 80)

    t0 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t0, 1, (30, 30, 20), size=5)
    t1 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t1, 11, (30, 30, 24), size=5)
    t2 = np.zeros(shape, dtype=np.int32)
    _paint_cube(t2, 12, (30, 30, 29), size=5)

    tracker = PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
        max_misses=10,
        gating_params={
            "kalman_brake_velocity_decay": 0.4,
            "kalman_brake_acceleration_decay": 0.1,
            "kalman_freeze_after_misses": 3,
        },
    )
    tracker.set_reference(_make_snapshot(t0, 0))
    tracker.track_snapshot(_make_snapshot(t1, 1))
    tracker.track_snapshot(_make_snapshot(t2, 2))
    tracker.track_snapshot(_make_snapshot(np.zeros(shape, dtype=np.int32), 3))
    tracker.track_snapshot(_make_snapshot(np.zeros(shape, dtype=np.int32), 4))
    tracker.track_snapshot(_make_snapshot(np.zeros(shape, dtype=np.int32), 5))
    tracker.track_snapshot(_make_snapshot(np.zeros(shape, dtype=np.int32), 6))

    history = np.asarray(tracker.get_results().tracking.center_history[1], dtype=np.float64)
    x = history[:, 2]
    d1 = abs(float(x[3] - x[2]))
    d2 = abs(float(x[4] - x[3]))
    d3 = abs(float(x[5] - x[4]))
    d4 = abs(float(x[6] - x[5]))

    assert d2 <= d1 + 1e-6
    assert d3 <= d2 + 1e-6
    assert d4 <= 1e-6
