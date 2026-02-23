import numpy as np

from core.coordinates import (
    voxel_delta_zyx_to_world_delta_xyz,
    world_delta_xyz_to_voxel_delta_zyx,
)
from processors.tracking_utils import extract_shifted_overlap_region


def test_world_voxel_delta_roundtrip_anisotropic():
    spacing = (2.0, 3.0, 4.0)  # x, y, z
    delta_xyz = (6.0, -9.0, 8.0)

    delta_zyx = world_delta_xyz_to_voxel_delta_zyx(delta_xyz=delta_xyz, spacing_xyz=spacing)
    assert np.allclose(delta_zyx, np.asarray((2.0, -3.0, 3.0), dtype=np.float64))

    roundtrip_xyz = voxel_delta_zyx_to_world_delta_xyz(delta_zyx=tuple(delta_zyx.tolist()), spacing_xyz=spacing)
    assert np.allclose(roundtrip_xyz, np.asarray(delta_xyz, dtype=np.float64))


def test_extract_shifted_overlap_region_uses_explicit_voxel_shift():
    current = np.zeros((8, 8, 8), dtype=np.int32)
    local_ref_mask = np.zeros((2, 2, 2), dtype=bool)
    local_ref_mask[0, 0, 0] = True

    bbox_mins = np.asarray((2, 3, 1), dtype=np.int64)
    shift_zyx = np.asarray((1, -2, 3), dtype=np.float64)  # will be rounded in helper
    target_mins = bbox_mins + np.asarray((1, -2, 3), dtype=np.int64)
    current[target_mins[0], target_mins[1], target_mins[2]] = 7

    shifted_ref_mask, shifted_current_region = extract_shifted_overlap_region(
        current_regions=current,
        local_reference_mask=local_ref_mask,
        bbox_mins=bbox_mins,
        shift_zyx=shift_zyx,
    )

    assert shifted_ref_mask.shape == (2, 2, 2)
    assert shifted_current_region.shape == (2, 2, 2)
    assert bool(shifted_ref_mask[0, 0, 0]) is True
    assert int(shifted_current_region[0, 0, 0]) == 7
