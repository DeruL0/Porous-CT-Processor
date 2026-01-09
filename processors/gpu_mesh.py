"""
GPU-accelerated mesh generation for throat tubes.
Generates cylinder vertices and faces directly on GPU using CuPy.
"""

import numpy as np
from typing import Tuple, List
import time

from config import GPU_ENABLED
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def generate_tubes_mesh(connections: List[Tuple], n_sides: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tube mesh for all throat connections.
    
    Args:
        connections: List of (p1, p2, radius) tuples
        n_sides: Number of sides for each tube (6 = hexagonal, fast)
        
    Returns:
        (vertices, faces) - NumPy arrays for PyVista mesh creation
    """
    if not connections:
        return np.array([]), np.array([])
    
    n_tubes = len(connections)
    print(f"[Mesh] Generating {n_tubes} tubes ({n_sides} sides each)...")
    start = time.time()
    
    backend = get_gpu_backend()
    
    if GPU_ENABLED and backend.available and CUPY_AVAILABLE and n_tubes > 100:
        try:
            vertices, faces = _generate_tubes_gpu(connections, n_sides)
            elapsed = time.time() - start
            print(f"[Mesh] GPU completed: {elapsed:.2f}s, {len(vertices)} vertices")
            return vertices, faces
        except Exception as e:
            print(f"[Mesh] GPU failed ({e}), using CPU")
    
    # CPU fallback
    vertices, faces = _generate_tubes_cpu(connections, n_sides)
    elapsed = time.time() - start
    print(f"[Mesh] CPU completed: {elapsed:.2f}s, {len(vertices)} vertices")
    return vertices, faces


def _generate_tubes_gpu(connections: List[Tuple], n_sides: int) -> Tuple[np.ndarray, np.ndarray]:
    """GPU implementation of tube mesh generation."""
    n_tubes = len(connections)
    
    # Extract connection data
    p1_arr = np.array([c[0] for c in connections], dtype=np.float32)  # (N, 3)
    p2_arr = np.array([c[1] for c in connections], dtype=np.float32)  # (N, 3)
    radii = np.array([c[2] for c in connections], dtype=np.float32)   # (N,)
    
    # Transfer to GPU
    p1_gpu = cp.asarray(p1_arr)
    p2_gpu = cp.asarray(p2_arr)
    radii_gpu = cp.asarray(radii)
    
    # Compute direction vectors
    direction = p2_gpu - p1_gpu  # (N, 3)
    length = cp.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / (length + 1e-10)  # Normalize
    
    # Compute perpendicular vectors using cross product
    # Use [0, 0, 1] as reference, or [0, 1, 0] if parallel
    up = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
    perp1 = cp.cross(direction, up)
    perp1_norm = cp.linalg.norm(perp1, axis=1, keepdims=True)
    
    # Handle cases where direction is parallel to up
    parallel_mask = perp1_norm.flatten() < 0.1
    alt_up = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)
    perp1_alt = cp.cross(direction, alt_up)
    perp1[parallel_mask] = perp1_alt[parallel_mask]
    perp1_norm = cp.linalg.norm(perp1, axis=1, keepdims=True)
    perp1 = perp1 / (perp1_norm + 1e-10)
    
    # Second perpendicular
    perp2 = cp.cross(direction, perp1)
    perp2 = perp2 / (cp.linalg.norm(perp2, axis=1, keepdims=True) + 1e-10)
    
    # Generate vertices for all tubes
    # Each tube has 2 rings of n_sides vertices
    angles = cp.linspace(0, 2 * cp.pi, n_sides, endpoint=False, dtype=cp.float32)
    cos_angles = cp.cos(angles)  # (S,)
    sin_angles = cp.sin(angles)  # (S,)
    
    # Broadcast to compute all vertices at once
    # Shape: (N, S, 3) for each ring
    radii_exp = radii_gpu[:, None, None]  # (N, 1, 1)
    perp1_exp = perp1[:, None, :]  # (N, 1, 3)
    perp2_exp = perp2[:, None, :]  # (N, 1, 3)
    cos_exp = cos_angles[None, :, None]  # (1, S, 1)
    sin_exp = sin_angles[None, :, None]  # (1, S, 1)
    
    offset = radii_exp * (cos_exp * perp1_exp + sin_exp * perp2_exp)  # (N, S, 3)
    
    ring1 = p1_gpu[:, None, :] + offset  # (N, S, 3)
    ring2 = p2_gpu[:, None, :] + offset  # (N, S, 3)
    
    # Combine rings: (N, 2, S, 3) -> (N * 2 * S, 3)
    vertices_gpu = cp.concatenate([ring1, ring2], axis=1)  # (N, 2*S, 3)
    vertices_gpu = vertices_gpu.reshape(-1, 3)
    
    # Generate faces (triangles)
    # Each side of each tube has 2 triangles
    # Vertices layout per tube: [ring1_0, ring1_1, ..., ring1_S-1, ring2_0, ring2_1, ..., ring2_S-1]
    verts_per_tube = 2 * n_sides
    faces_list = []
    
    for i in range(n_sides):
        # Indices within one tube
        i0 = i              # ring1, current
        i1 = (i + 1) % n_sides  # ring1, next
        i2 = n_sides + i    # ring2, current
        i3 = n_sides + (i + 1) % n_sides  # ring2, next
        
        # Triangle 1: i0, i2, i1
        # Triangle 2: i1, i2, i3
        faces_list.append([i0, i2, i1])
        faces_list.append([i1, i2, i3])
    
    face_template = cp.array(faces_list, dtype=cp.int32)  # (n_sides * 2, 3)
    
    # Broadcast to all tubes
    tube_offsets = cp.arange(n_tubes, dtype=cp.int32)[:, None, None] * verts_per_tube
    faces_gpu = face_template[None, :, :] + tube_offsets  # (N, n_sides*2, 3)
    faces_gpu = faces_gpu.reshape(-1, 3)
    
    # Transfer to CPU
    vertices = cp.asnumpy(vertices_gpu)
    faces_indices = cp.asnumpy(faces_gpu)
    
    # Convert to PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = len(faces_indices)
    faces = np.empty((n_faces, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = faces_indices
    faces = faces.flatten()
    
    return vertices, faces


def _generate_tubes_cpu(connections: List[Tuple], n_sides: int) -> Tuple[np.ndarray, np.ndarray]:
    """CPU implementation of tube mesh generation."""
    n_tubes = len(connections)
    
    # Pre-allocate arrays
    verts_per_tube = 2 * n_sides
    faces_per_tube = 2 * n_sides  # 2 triangles per side
    
    all_vertices = np.zeros((n_tubes * verts_per_tube, 3), dtype=np.float32)
    all_faces = np.zeros((n_tubes * faces_per_tube, 4), dtype=np.int64)
    
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    up = np.array([0.0, 0.0, 1.0])
    alt_up = np.array([0.0, 1.0, 0.0])
    
    for idx, (p1, p2, radius) in enumerate(connections):
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # Direction
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-10:
            continue
        direction = direction / length
        
        # Perpendicular vectors
        perp1 = np.cross(direction, up)
        if np.linalg.norm(perp1) < 0.1:
            perp1 = np.cross(direction, alt_up)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Generate ring vertices
        v_start = idx * verts_per_tube
        for i in range(n_sides):
            offset = radius * (cos_angles[i] * perp1 + sin_angles[i] * perp2)
            all_vertices[v_start + i] = p1 + offset
            all_vertices[v_start + n_sides + i] = p2 + offset
        
        # Generate faces
        f_start = idx * faces_per_tube
        for i in range(n_sides):
            i0 = v_start + i
            i1 = v_start + (i + 1) % n_sides
            i2 = v_start + n_sides + i
            i3 = v_start + n_sides + (i + 1) % n_sides
            
            all_faces[f_start + i * 2] = [3, i0, i2, i1]
            all_faces[f_start + i * 2 + 1] = [3, i1, i2, i3]
    
    return all_vertices, all_faces.flatten()
