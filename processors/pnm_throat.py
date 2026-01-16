"""
Throat mesh generation algorithms for Pore Network Modeling.
GPU-accelerated batch tube generation with CPU fallback.
"""

import numpy as np
import scipy.ndimage as ndimage
import pyvista as pv
from typing import Tuple, List, Optional, Set, Dict
from concurrent.futures import ThreadPoolExecutor

from config import GPU_ENABLED
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE


def create_throat_mesh(
    connections: Set[Tuple[int, int]],
    centers: np.ndarray,
    radii: np.ndarray, 
    ids_list: np.ndarray,
    distance_map: Optional[np.ndarray] = None,
    segmented_regions: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[pv.PolyData, np.ndarray]:
    """
    Create throat mesh connecting pore centers.
    
    Uses accurate distance-transform-based throat radius measurement when
    distance_map and segmented_regions are provided.
    
    Args:
        connections: Set of (id_a, id_b) tuples indicating connected pores
        centers: Array of pore center coordinates (N, 3)
        radii: Array of pore radii (N,)
        ids_list: Array of pore IDs (N,)
        distance_map: Optional distance transform for accurate throat measurement
        segmented_regions: Optional segmented labels for throat measurement
        spacing: Voxel spacing (x, y, z)
        
    Returns:
        Tuple of (throat_mesh, throat_radii_array)
    """
    if not connections or len(centers) == 0:
        return pv.PolyData(), np.array([])

    id_map = {uid: idx for idx, uid in enumerate(ids_list)}
    
    use_accurate = (distance_map is not None and 
                    segmented_regions is not None and 
                    spacing is not None)

    # Pre-compute bounding boxes for throat measurement
    throat_radii_map = {}
    if use_accurate:
        slices = ndimage.find_objects(segmented_regions)
        slices_cache = {i + 1: s for i, s in enumerate(slices) if s is not None}
        
        # Parallel throat radius measurement
        def measure_throat(conn):
            id_a, id_b = conn
            key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            radius, _ = _measure_throat_radius_fast(
                distance_map, segmented_regions, id_a, id_b, spacing, slices_cache
            )
            return key, radius
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(measure_throat, connections))
        
        for key, radius in results:
            if radius > 0:
                throat_radii_map[key] = radius

    # Build valid connections
    valid_connections = []
    for id_a, id_b in connections:
        if id_a not in id_map or id_b not in id_map:
            continue
            
        idx_a = id_map[id_a]
        idx_b = id_map[id_b]
        p1 = centers[idx_a]
        p2 = centers[idx_b]
        
        key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
        r_throat = throat_radii_map.get(key, 0.0)
        
        if r_throat <= 0:
            r_throat = min(radii[idx_a], radii[idx_b]) * 0.3
        
        valid_connections.append((p1, p2, r_throat))

    if not valid_connections:
        return pv.PolyData(), np.array([])

    throat_radii = np.array([c[2] for c in valid_connections])

    # GPU batch tube generation
    backend = get_gpu_backend()
    if GPU_ENABLED and backend.available and CUPY_AVAILABLE and len(valid_connections) > 50:
        try:
            vertices, faces = _generate_tubes_batch_gpu(valid_connections, n_sides=6)
            if len(vertices) > 0:
                throats = pv.PolyData(vertices, faces)
                throats["IsPore"] = np.zeros(throats.n_points, dtype=int)
                print(f"[GPU] Generated {len(valid_connections)} tubes")
                return throats, throat_radii
        except Exception as e:
            print(f"[GPU] Tube generation failed: {e}, using CPU")
            backend.clear_memory()

    # CPU fallback - batch with PyVista
    throats = _generate_tubes_batch_cpu(valid_connections)
    throats["IsPore"] = np.zeros(throats.n_points, dtype=int)
    
    return throats, throat_radii


def _generate_tubes_batch_gpu(connections: List[Tuple], n_sides: int = 6):
    """Generate all tube meshes on GPU, transfer once at end."""
    # Pre-compute tube template (unit cylinder along Z)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for p1, p2, radius in connections:
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # Direction and length
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length
        
        # Build rotation matrix to align Z with direction
        z_axis = np.array([0, 0, 1])
        if np.abs(np.dot(direction, z_axis)) > 0.999:
            # Nearly parallel to Z
            rotation = np.eye(3) if direction[2] > 0 else np.diag([1, -1, -1])
        else:
            v = np.cross(z_axis, direction)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, direction)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        
        # Generate tube vertices
        # Bottom ring
        bottom_ring = np.column_stack([
            circle_x * radius * 0.5,
            circle_y * radius * 0.5,
            np.zeros(n_sides)
        ])
        # Top ring
        top_ring = np.column_stack([
            circle_x * radius * 0.5,
            circle_y * radius * 0.5,
            np.ones(n_sides) * length
        ])
        
        # Transform
        bottom_transformed = (rotation @ bottom_ring.T).T + p1
        top_transformed = (rotation @ top_ring.T).T + p1
        
        tube_vertices = np.vstack([bottom_transformed, top_transformed])
        all_vertices.append(tube_vertices)
        
        # Generate faces (quads as two triangles)
        tube_faces = []
        for i in range(n_sides):
            next_i = (i + 1) % n_sides
            # Bottom triangle
            tube_faces.append([3, vertex_offset + i, vertex_offset + next_i, vertex_offset + n_sides + i])
            # Top triangle
            tube_faces.append([3, vertex_offset + next_i, vertex_offset + n_sides + next_i, vertex_offset + n_sides + i])
        
        all_faces.extend(tube_faces)
        vertex_offset += 2 * n_sides
    
    if not all_vertices:
        return np.array([]), np.array([])
    
    vertices = np.vstack(all_vertices).astype(np.float32)
    faces = np.hstack([[item for sublist in all_faces for item in sublist]]).astype(np.int32)
    
    return vertices, faces


def _generate_tubes_batch_cpu(connections: List[Tuple]) -> pv.PolyData:
    """Generate tubes using PyVista (CPU fallback)."""
    tubes_list = []
    
    for p1, p2, r_throat in connections:
        try:
            line = pv.Line(p1, p2)
            tube = line.tube(radius=max(r_throat * 0.5, 0.1), n_sides=6)
            tubes_list.append(tube)
        except Exception:
            continue
    
    if not tubes_list:
        return pv.PolyData()
    
    # Merge all tubes
    result = tubes_list[0]
    for tube in tubes_list[1:]:
        result = result.merge(tube)
    
    return result


def _measure_throat_radius_fast(
    distance_map: np.ndarray,
    segmented_regions: np.ndarray, 
    pore_a: int, 
    pore_b: int, 
    spacing: Tuple[float, float, float],
    slices_cache: Dict[int, tuple] = None
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Measure throat radius using bounding box optimization.
    
    Uses distance transform values at the boundary between two pores.
    
    Args:
        distance_map: Euclidean distance transform of pore space
        segmented_regions: Labeled regions from watershed
        pore_a, pore_b: Pore label IDs
        spacing: Voxel spacing
        slices_cache: Pre-computed slice objects for each pore
        
    Returns:
        Tuple of (throat_radius, None)
    """
    if slices_cache is not None:
        slice_a = slices_cache.get(pore_a)
        slice_b = slices_cache.get(pore_b)
    else:
        return 0.0, None
    
    if slice_a is None or slice_b is None:
        return 0.0, None
    
    # Compute union bounding box
    shape = segmented_regions.shape
    z_min = max(0, min(slice_a[0].start, slice_b[0].start) - 1)
    z_max = min(shape[0], max(slice_a[0].stop, slice_b[0].stop) + 1)
    y_min = max(0, min(slice_a[1].start, slice_b[1].start) - 1)
    y_max = min(shape[1], max(slice_a[1].stop, slice_b[1].stop) + 1)
    x_min = max(0, min(slice_a[2].start, slice_b[2].start) - 1)
    x_max = min(shape[2], max(slice_a[2].stop, slice_b[2].stop) + 1)
    
    # Extract local region
    local_labels = segmented_regions[z_min:z_max, y_min:y_max, x_min:x_max]
    local_dist = distance_map[z_min:z_max, y_min:y_max, x_min:x_max]
    
    mask_a = (local_labels == pore_a)
    mask_b = (local_labels == pore_b)
    
    # Find boundary using dilation
    dilated_a = ndimage.binary_dilation(mask_a)
    dilated_b = ndimage.binary_dilation(mask_b)
    
    throat_region = dilated_a & dilated_b
    
    if not np.any(throat_region):
        return 0.0, None
    
    throat_distances = local_dist[throat_region]
    
    if len(throat_distances) == 0:
        return 0.0, None
    
    throat_radius_voxels = float(np.min(throat_distances))
    avg_spacing = (spacing[0] + spacing[1] + spacing[2]) / 3.0
    throat_radius = throat_radius_voxels * avg_spacing
    
    return throat_radius, None
