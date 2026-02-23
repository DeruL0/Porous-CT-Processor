import numpy as np

from core.time_series import PNMSnapshot
from processors.pnm_tracker import PNMTracker
from processors.tracking_utils import MacroRegistrationResult


def _make_single_pore_snapshot(
    pore_id: int = 1,
    center_xyz=(10.0, 20.0, 30.0),
    spacing=(2.0, 3.0, 4.0),
    origin=(0.0, 0.0, 0.0),
) -> PNMSnapshot:
    regions = np.zeros((8, 8, 8), dtype=np.int32)
    regions[2:4, 2:4, 2:4] = int(pore_id)
    return PNMSnapshot(
        time_index=0,
        pore_centers=np.asarray([center_xyz], dtype=np.float64),
        pore_radii=np.asarray([1.0], dtype=np.float64),
        pore_ids=np.asarray([int(pore_id)], dtype=np.int32),
        pore_volumes=np.asarray([8.0], dtype=np.float64),
        connections=[],
        segmented_regions=regions,
        spacing=spacing,
        origin=origin,
        metadata={},
    )


def test_compose_expected_center_converts_macro_voxel_shift_to_world_delta():
    tracker = PNMTracker(match_mode="temporal_global", assign_solver="scipy", use_gpu=False, use_batch=False)
    ref = _make_single_pore_snapshot(center_xyz=(10.0, 20.0, 30.0), spacing=(2.0, 3.0, 4.0))
    tracker.set_reference(ref)

    macro = MacroRegistrationResult(
        displacement=np.asarray((1.0, 2.0, 3.0), dtype=np.float64),  # z, y, x voxels
        confidence=1.0,
        method="unit_test",
    )
    out = tracker._compose_expected_center(
        ref_id=1,
        motion_prediction=np.asarray((10.0, 20.0, 30.0), dtype=np.float64),
        macro_registration=macro,
    )

    # delta world xyz = (dx*sx, dy*sy, dz*sz) = (3*2, 2*3, 1*4)
    assert np.allclose(out, np.asarray((16.0, 26.0, 34.0), dtype=np.float64))


def test_closure_sampling_uses_world_to_voxel_coordinates():
    tracker = PNMTracker(match_mode="temporal_global", assign_solver="scipy", use_gpu=False, use_batch=False)
    ref = _make_single_pore_snapshot(center_xyz=(0.0, 0.0, 0.0), spacing=(2.0, 3.0, 4.0), origin=(0.0, 0.0, 0.0))
    tracker.set_reference(ref)

    class _DummyMacro:
        def __init__(self):
            self.last_point = None

        def sample_compression(self, point_zyx, default=0.0):
            self.last_point = np.asarray(point_zyx, dtype=np.float64)
            return 0.0

    macro = _DummyMacro()
    match_results = {1: {"matched_id": -1, "reason": "gate_rejected"}}
    predicted_centers = {1: np.asarray((2.0, 6.0, 12.0), dtype=np.float64)}  # xyz world

    tracker._apply_closure_classification(
        reference_snapshot=ref,
        match_results=match_results,
        predicted_centers=predicted_centers,
        macro_registration=macro,  # duck-typed
    )

    # xyz -> zyx voxel index with spacing (2,3,4): (z=3, y=2, x=1)
    assert macro.last_point is not None
    assert np.allclose(macro.last_point, np.asarray((3.0, 2.0, 1.0), dtype=np.float64))
