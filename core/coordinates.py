"""
Coordinate conversion helpers for the project-wide 3D convention.

Convention:
- Raw voxel arrays use index order (z, y, x)
- World-space geometry uses axis order (x, y, z)
- Spacing/origin tuples are stored as (x, y, z)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def raw_zyx_to_grid_xyz(raw_data: np.ndarray) -> np.ndarray:
    """
    Reorder a raw volume from (z, y, x) to (x, y, z) for VTK/PyVista grids.
    """
    arr = np.asarray(raw_data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr.shape}")
    return np.transpose(arr, (2, 1, 0))


def world_xyz_to_voxel_zyx(
    world_xyz: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Convert world coordinates (x, y, z) to fractional voxel indices (z, y, x).
    """
    xw, yw, zw = world_xyz
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    if abs(sx) < 1e-12 or abs(sy) < 1e-12 or abs(sz) < 1e-12:
        raise ValueError("Spacing components must be non-zero.")
    x_idx = (xw - ox) / sx
    y_idx = (yw - oy) / sy
    z_idx = (zw - oz) / sz
    return (z_idx, y_idx, x_idx)


def world_xyz_to_index_zyx(
    world_xyz: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    *,
    rounding: str = "round",
) -> Tuple[int, int, int]:
    """
    Convert world coordinates (x, y, z) to integer voxel indices (z, y, x).

    rounding:
    - "round" (default): nearest integer
    - "floor": floor toward -inf
    - "ceil": ceil toward +inf
    """
    zf, yf, xf = world_xyz_to_voxel_zyx(world_xyz, spacing_xyz, origin_xyz)

    mode = str(rounding).strip().lower()
    if mode == "round":
        return (int(np.rint(zf)), int(np.rint(yf)), int(np.rint(xf)))
    if mode == "floor":
        return (int(np.floor(zf)), int(np.floor(yf)), int(np.floor(xf)))
    if mode == "ceil":
        return (int(np.ceil(zf)), int(np.ceil(yf)), int(np.ceil(xf)))
    raise ValueError(f"Unknown rounding mode: {rounding}")


def world_delta_xyz_to_voxel_delta_zyx(
    delta_xyz: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """
    Convert a world-space delta vector (x, y, z) into voxel-space delta (z, y, x).
    """
    dx, dy, dz = (float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2]))
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    if abs(sx) < 1e-12 or abs(sy) < 1e-12 or abs(sz) < 1e-12:
        raise ValueError("Spacing components must be non-zero.")
    return np.asarray((dz / sz, dy / sy, dx / sx), dtype=np.float64)


def voxel_delta_zyx_to_world_delta_xyz(
    delta_zyx: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """
    Convert a voxel-space delta (z, y, x) into world-space delta (x, y, z).
    """
    dz, dy, dx = (float(delta_zyx[0]), float(delta_zyx[1]), float(delta_zyx[2]))
    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    return np.asarray((dx * sx, dy * sy, dz * sz), dtype=np.float64)


def bounds_xyz_to_slices_zyx(
    bounds_xyz: Tuple[float, float, float, float, float, float],
    raw_shape_zyx: Tuple[int, int, int],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> Tuple[int, int, int, int, int, int]:
    """
    Convert world bounds (xmin, xmax, ymin, ymax, zmin, zmax)
    into clamped voxel slice bounds (z0, z1, y0, y1, x0, x1).
    """
    x0, x1, y0, y1, z0, z1 = bounds_xyz
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if z0 > z1:
        z0, z1 = z1, z0

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz

    x_start = int(np.floor((x0 - ox) / sx))
    x_end = int(np.floor((x1 - ox) / sx))
    y_start = int(np.floor((y0 - oy) / sy))
    y_end = int(np.floor((y1 - oy) / sy))
    z_start = int(np.floor((z0 - oz) / sz))
    z_end = int(np.floor((z1 - oz) / sz))

    z_start = max(0, min(raw_shape_zyx[0], z_start))
    z_end = max(0, min(raw_shape_zyx[0], z_end))
    y_start = max(0, min(raw_shape_zyx[1], y_start))
    y_end = max(0, min(raw_shape_zyx[1], y_end))
    x_start = max(0, min(raw_shape_zyx[2], x_start))
    x_end = max(0, min(raw_shape_zyx[2], x_end))
    return (z_start, z_end, y_start, y_end, x_start, x_end)


def origin_xyz_for_subvolume_zyx(
    origin_xyz: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
    z_start: int,
    y_start: int,
    x_start: int,
) -> Tuple[float, float, float]:
    """
    Compute world origin for a cropped subvolume indexed in (z, y, x).
    """
    ox, oy, oz = origin_xyz
    sx, sy, sz = spacing_xyz
    return (
        ox + x_start * sx,
        oy + y_start * sy,
        oz + z_start * sz,
    )


def voxel_grid_zyx_to_world_xyz(
    z_idx: np.ndarray,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert voxel-index grids (z, y, x) to world-coordinate grids (x, y, z).
    """
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    world_x = ox + x_idx * sx
    world_y = oy + y_idx * sy
    world_z = oz + z_idx * sz
    return world_x, world_y, world_z


def voxel_zyx_to_world_xyz(
    z_idx: float,
    y_idx: float,
    x_idx: float,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Convert one voxel index triple (z, y, x) to one world coordinate (x, y, z).
    """
    world_x, world_y, world_z = voxel_grid_zyx_to_world_xyz(
        np.asarray(z_idx),
        np.asarray(y_idx),
        np.asarray(x_idx),
        spacing_xyz,
        origin_xyz,
    )
    return (float(world_x), float(world_y), float(world_z))
