"""
ROI shape extraction algorithms for volumetric data.
Provides box, ellipsoid, and cylinder extraction from 3D arrays.
"""

from typing import Optional, Tuple
import numpy as np

from core import VolumeData


def bounds_to_voxel_indices(
    bounds: tuple, 
    raw_shape: tuple, 
    spacing: tuple, 
    origin: tuple
) -> Tuple[int, int, int, int, int, int]:
    """
    Convert world coordinate bounds to voxel array indices.
    
    Args:
        bounds: World coordinate bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        raw_shape: Shape of the raw data array (z, y, x)
        spacing: Voxel spacing (x, y, z)
        origin: World coordinate origin (x, y, z)
        
    Returns:
        Tuple of (i_start, i_end, j_start, j_end, k_start, k_end)
    """
    i_start = max(0, int((bounds[0] - origin[0]) / spacing[0]))
    i_end = min(raw_shape[0], int((bounds[1] - origin[0]) / spacing[0]))
    j_start = max(0, int((bounds[2] - origin[1]) / spacing[1]))
    j_end = min(raw_shape[1], int((bounds[3] - origin[1]) / spacing[1]))
    k_start = max(0, int((bounds[4] - origin[2]) / spacing[2]))
    k_end = min(raw_shape[2], int((bounds[5] - origin[2]) / spacing[2]))
    
    return i_start, i_end, j_start, j_end, k_start, k_end


def _calculate_new_origin(
    origin: tuple, 
    spacing: tuple, 
    i_start: int, 
    j_start: int, 
    k_start: int
) -> Tuple[float, float, float]:
    """Calculate origin for extracted sub-volume."""
    return (
        origin[0] + i_start * spacing[0],
        origin[1] + j_start * spacing[1],
        origin[2] + k_start * spacing[2]
    )


def _build_result(
    sub_data: np.ndarray,
    spacing: tuple,
    new_origin: tuple,
    base_metadata: dict,
    type_label: str,
    bounds: tuple,
    extra_metadata: dict = None
) -> Optional[VolumeData]:
    """Build VolumeData result with metadata."""
    if sub_data.size == 0:
        return None
    
    metadata = dict(base_metadata)
    metadata['Type'] = type_label
    metadata['ROI_Bounds'] = bounds
    if extra_metadata:
        metadata.update(extra_metadata)
    
    return VolumeData(raw_data=sub_data, spacing=spacing, origin=new_origin, metadata=metadata)


def extract_box(
    data: VolumeData, 
    bounds: tuple
) -> Optional[VolumeData]:
    """
    Extract box-shaped sub-volume from data.
    
    Args:
        data: Input VolumeData
        bounds: World coordinate bounds
        
    Returns:
        Extracted VolumeData with new origin, or None if extraction fails
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin
    
    i_start, i_end, j_start, j_end, k_start, k_end = bounds_to_voxel_indices(
        bounds, raw.shape, spacing, origin
    )
    sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
    
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, i_start, j_start, k_start)
    return _build_result(
        sub_data, spacing, new_origin, data.metadata,
        f"ROI Box ({sub_data.shape})", bounds
    )


def extract_ellipsoid(
    data: VolumeData, 
    bounds: tuple
) -> Optional[VolumeData]:
    """
    Extract ellipsoid sub-volume inscribed in box bounds.
    
    Areas outside the ellipsoid are set to the minimum value.
    
    Args:
        data: Input VolumeData
        bounds: World coordinate bounds
        
    Returns:
        Extracted VolumeData with ellipsoid mask applied
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin
    
    i_start, i_end, j_start, j_end, k_start, k_end = bounds_to_voxel_indices(
        bounds, raw.shape, spacing, origin
    )
    
    # Calculate center and radii
    rx = (bounds[1] - bounds[0]) / 2
    ry = (bounds[3] - bounds[2]) / 2
    rz = (bounds[5] - bounds[4]) / 2
    
    ci = ((bounds[0] + bounds[1]) / 2 - origin[0]) / spacing[0]
    cj = ((bounds[2] + bounds[3]) / 2 - origin[1]) / spacing[1]
    ck = ((bounds[4] + bounds[5]) / 2 - origin[2]) / spacing[2]
    
    # Create ellipsoid mask
    ii, jj, kk = np.meshgrid(
        np.arange(i_start, i_end),
        np.arange(j_start, j_end),
        np.arange(k_start, k_end),
        indexing='ij'
    )
    
    dist = np.sqrt(
        ((ii - ci) * spacing[0] / rx) ** 2 +
        ((jj - cj) * spacing[1] / ry) ** 2 +
        ((kk - ck) * spacing[2] / rz) ** 2
    )
    mask = dist <= 1.0
    
    sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
    sub_data[~mask] = sub_data.min()
    
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, i_start, j_start, k_start)
    return _build_result(
        sub_data, spacing, new_origin, data.metadata,
        f"ROI Ellipsoid ({sub_data.shape})", bounds,
        {'ROI_Radii': (rx, ry, rz)}
    )


def extract_cylinder(
    data: VolumeData, 
    bounds: tuple,
    current_size: Optional[Tuple[float, float, float]] = None,
    transform: Optional[np.ndarray] = None
) -> Optional[VolumeData]:
    """
    Extract elliptical cylinder inscribed in box bounds.
    
    Supports rotation via transform matrix.
    
    Args:
        data: Input VolumeData
        bounds: World coordinate bounds (AABB)
        current_size: Pre-rotation size (x, y, z) if available
        transform: 4x4 transformation matrix containing rotation
        
    Returns:
        Extracted VolumeData with cylinder mask applied
    """
    if data is None or data.raw_data is None:
        return None

    raw = data.raw_data
    spacing = data.spacing
    origin = data.origin
    
    i_start, i_end, j_start, j_end, k_start, k_end = bounds_to_voxel_indices(
        bounds, raw.shape, spacing, origin
    )
    
    # Use stored size if available (rotation-invariant)
    if current_size:
        size_x, size_y, size_z = current_size
    else:
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]
    
    # Radii for elliptical cross-section
    rx = size_x / 2  # Cylinder length axis
    ry = size_y / 2
    rz = size_z / 2
    
    # Center in world coordinates (from AABB)
    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ])
    
    # Create voxel coordinate grids
    ii, jj, kk = np.meshgrid(
        np.arange(i_start, i_end),
        np.arange(j_start, j_end),
        np.arange(k_start, k_end),
        indexing='ij'
    )
    
    # Convert to world coordinates
    world_x = origin[0] + ii * spacing[0]
    world_y = origin[1] + jj * spacing[1]
    world_z = origin[2] + kk * spacing[2]
    
    # Get relative position to center
    rel_x = world_x - center[0]
    rel_y = world_y - center[1]
    rel_z = world_z - center[2]
    
    # Apply inverse rotation if transform exists
    if transform is not None:
        # Extract rotation matrix and normalize (remove scale)
        rotation = transform[:3, :3].copy()
        for i in range(3):
            col_norm = np.linalg.norm(rotation[:, i])
            if col_norm > 1e-6:
                rotation[:, i] /= col_norm
        
        # Inverse rotation = transpose for orthogonal matrix
        inv_rotation = rotation.T
        
        # Transform relative coordinates to local space
        local_x = inv_rotation[0, 0] * rel_x + inv_rotation[0, 1] * rel_y + inv_rotation[0, 2] * rel_z
        local_y = inv_rotation[1, 0] * rel_x + inv_rotation[1, 1] * rel_y + inv_rotation[1, 2] * rel_z
        local_z = inv_rotation[2, 0] * rel_x + inv_rotation[2, 1] * rel_y + inv_rotation[2, 2] * rel_z
    else:
        local_x = rel_x
        local_y = rel_y
        local_z = rel_z
    
    # Cylinder mask: inside if (y/ry)^2 + (z/rz)^2 <= 1 AND abs(x) <= rx
    dist_2d = np.sqrt((local_y / ry) ** 2 + (local_z / rz) ** 2)
    mask = (dist_2d <= 1.0) & (np.abs(local_x) <= rx)
    
    sub_data = raw[i_start:i_end, j_start:j_end, k_start:k_end].copy()
    sub_data[~mask] = sub_data.min()
    
    if sub_data.size == 0:
        return None

    new_origin = _calculate_new_origin(origin, spacing, i_start, j_start, k_start)
    extra = {'ROI_Radii_YZ': (ry, rz)}
    if transform is not None:
        extra['ROI_Rotated'] = True
    
    return _build_result(
        sub_data, spacing, new_origin, data.metadata,
        f"ROI Elliptical Cylinder ({sub_data.shape})", bounds, extra
    )

