"""
ROI shape extraction algorithms for volumetric data.
Provides box, ellipsoid, and cylinder extraction from 3D arrays.
"""

from typing import Optional, Tuple
import numpy as np

from core import VolumeData
from core.coordinates import (
    bounds_xyz_to_slices_zyx,
    origin_xyz_for_subvolume_zyx,
    voxel_grid_zyx_to_world_xyz,
    world_xyz_to_voxel_zyx,
)


def bounds_to_voxel_indices(
    bounds: tuple,
    raw_shape: tuple,
    spacing: tuple,
    origin: tuple,
) -> Tuple[int, int, int, int, int, int]:
    """
    Convert world coordinate bounds to voxel slices.

    Args:
        bounds: World coordinate bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        raw_shape: Shape of raw data (z, y, x)
        spacing: Voxel spacing (x, y, z)
        origin: World origin (x, y, z)

    Returns:
        (z_start, z_end, y_start, y_end, x_start, x_end)
    """
    return bounds_xyz_to_slices_zyx(bounds, raw_shape, spacing, origin)


def _calculate_new_origin(
    origin: tuple,
    spacing: tuple,
    z_start: int,
    y_start: int,
    x_start: int,
) -> Tuple[float, float, float]:
    """Calculate origin for extracted sub-volume."""
    return origin_xyz_for_subvolume_zyx(origin, spacing, z_start, y_start, x_start)


def _build_result(
    sub_data: np.ndarray,
    spacing: tuple,
    new_origin: tuple,
    base_metadata: dict,
    type_label: str,
    bounds: tuple,
    extra_metadata: Optional[dict] = None,
) -> Optional[VolumeData]:
    """Build VolumeData result with metadata."""
    if sub_data.size == 0:
        return None

    metadata = dict(base_metadata)
    metadata["Type"] = type_label
    metadata["ROI_Bounds"] = bounds
    if extra_metadata:
        metadata.update(extra_metadata)

    return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=metadata)


def extract_box(
    data: VolumeData,
    bounds: tuple,
) -> Optional[VolumeData]:
    """
    Extract box-shaped sub-volume from data.
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin

    z_start, z_end, y_start, y_end, x_start, x_end = bounds_to_voxel_indices(
        bounds,
        raw.shape,
        spacing,
        origin,
    )
    sub_data = raw[z_start:z_end, y_start:y_end, x_start:x_end].copy()
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, z_start, y_start, x_start)
    return _build_result(
        sub_data,
        spacing,
        new_origin,
        data.metadata,
        f"ROI Box ({sub_data.shape})",
        bounds,
    )


def extract_ellipsoid(
    data: VolumeData,
    bounds: tuple,
) -> Optional[VolumeData]:
    """
    Extract ellipsoid sub-volume inscribed in box bounds.

    Areas outside the ellipsoid are set to the minimum value.
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin

    z_start, z_end, y_start, y_end, x_start, x_end = bounds_to_voxel_indices(
        bounds,
        raw.shape,
        spacing,
        origin,
    )

    rx = (bounds[1] - bounds[0]) / 2.0
    ry = (bounds[3] - bounds[2]) / 2.0
    rz = (bounds[5] - bounds[4]) / 2.0
    if rx <= 0 or ry <= 0 or rz <= 0:
        return None

    center_xyz = (
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    )
    center_z, center_y, center_x = world_xyz_to_voxel_zyx(center_xyz, spacing, origin)
    sx, sy, sz = spacing

    zz, yy, xx = np.meshgrid(
        np.arange(z_start, z_end),
        np.arange(y_start, y_end),
        np.arange(x_start, x_end),
        indexing="ij",
    )

    dist = np.sqrt(
        ((xx - center_x) * sx / rx) ** 2
        + ((yy - center_y) * sy / ry) ** 2
        + ((zz - center_z) * sz / rz) ** 2
    )
    mask = dist <= 1.0

    sub_data = raw[z_start:z_end, y_start:y_end, x_start:x_end].copy()
    sub_data[~mask] = sub_data.min()
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, z_start, y_start, x_start)
    return _build_result(
        sub_data,
        spacing,
        new_origin,
        data.metadata,
        f"ROI Ellipsoid ({sub_data.shape})",
        bounds,
        {"ROI_Radii": (rx, ry, rz)},
    )


def extract_cylinder(
    data: VolumeData,
    bounds: tuple,
    current_size: Optional[Tuple[float, float, float]] = None,
    transform: Optional[np.ndarray] = None,
) -> Optional[VolumeData]:
    """
    Extract elliptical cylinder inscribed in box bounds.

    Supports rotation via transform matrix.
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin

    z_start, z_end, y_start, y_end, x_start, x_end = bounds_to_voxel_indices(
        bounds,
        raw.shape,
        spacing,
        origin,
    )

    if current_size:
        size_x, size_y, size_z = current_size
    else:
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]

    rx = size_x / 2.0
    ry = size_y / 2.0
    rz = size_z / 2.0
    if rx <= 0 or ry <= 0 or rz <= 0:
        return None

    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ],
        dtype=np.float64,
    )

    zz, yy, xx = np.meshgrid(
        np.arange(z_start, z_end),
        np.arange(y_start, y_end),
        np.arange(x_start, x_end),
        indexing="ij",
    )

    world_x, world_y, world_z = voxel_grid_zyx_to_world_xyz(zz, yy, xx, spacing, origin)

    rel_x = world_x - center[0]
    rel_y = world_y - center[1]
    rel_z = world_z - center[2]

    if transform is not None:
        # Strip scale components and keep pure rotation.
        rotation = transform[:3, :3].copy()
        for col in range(3):
            col_norm = np.linalg.norm(rotation[:, col])
            if col_norm > 1e-6:
                rotation[:, col] /= col_norm

        inv_rotation = rotation.T
        local_x = inv_rotation[0, 0] * rel_x + inv_rotation[0, 1] * rel_y + inv_rotation[0, 2] * rel_z
        local_y = inv_rotation[1, 0] * rel_x + inv_rotation[1, 1] * rel_y + inv_rotation[1, 2] * rel_z
        local_z = inv_rotation[2, 0] * rel_x + inv_rotation[2, 1] * rel_y + inv_rotation[2, 2] * rel_z
    else:
        local_x = rel_x
        local_y = rel_y
        local_z = rel_z

    dist_2d = np.sqrt((local_y / ry) ** 2 + (local_z / rz) ** 2)
    mask = (dist_2d <= 1.0) & (np.abs(local_x) <= rx)

    sub_data = raw[z_start:z_end, y_start:y_end, x_start:x_end].copy()
    sub_data[~mask] = sub_data.min()
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, z_start, y_start, x_start)
    extra = {"ROI_Radii_YZ": (ry, rz)}
    if transform is not None:
        extra["ROI_Rotated"] = True

    return _build_result(
        sub_data,
        spacing,
        new_origin,
        data.metadata,
        f"ROI Elliptical Cylinder ({sub_data.shape})",
        bounds,
        extra,
    )
