"""
Pore to sphere processor for Pore Network Modeling (PNM).
GPU-accelerated with CuPy, memory-optimized with disk caching for large volumes.
"""

import os
import tempfile
import time
import numpy as np
import scipy.ndimage as ndimage
import pyvista as pv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Tuple, Set, Optional, Callable, List
import gc

from core import BaseProcessor, VolumeData
from processors.utils import binary_fill_holes, distance_transform_edt, find_local_maxima, watershed_gpu
from config import GPU_ENABLED
from core.gpu_backend import get_gpu_backend, CUPY_AVAILABLE

# cc3d for fast 3D connected component labeling
try:
    import cc3d
    HAS_CC3D = True
except ImportError:
    HAS_CC3D = False

# Optional high-performance libraries
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class PoreToSphereProcessor(BaseProcessor):
    """GPU-Optimized Pore Network Modeling (PNM)."""

    MIN_PEAK_DISTANCE = 6

    def __init__(self):
        super().__init__()
        self._cache = {}

    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None,
                threshold: int = -300) -> VolumeData:
        """
        Process volume data to generate PNM mesh.
        
        If input data is already processed pore data (from PoreExtractionProcessor),
        it will be reused directly, avoiding duplicate segmentation.
        """
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        def report(p, msg):
            print(f"[PNM] {msg}")
            if callback: callback(p, msg)

        from data.disk_cache import get_segmentation_cache
        
        # Use shared cache for segmentation
        seg_cache = get_segmentation_cache()
        volume_id = f"{id(data.raw_data)}_{threshold}"
        
        # Internal PNM cache for watershed results
        cache_key = (id(data.raw_data), threshold)
        cached_result = self._cache.get(cache_key)

        if cached_result:
            report(10, "Cache Hit! Reusing watershed results...")
            distance_map = cached_result['distance_map']
            segmented_regions = cached_result['labels']
            num_pores = cached_result['num_pores']
            report(80, "Skipped segmentation. Generating optimized mesh...")
        else:
            # Check shared segmentation cache first
            cached_mask = seg_cache.get_pores_mask(volume_id)
            
            if cached_mask is not None:
                report(0, "Using cached pores_mask from shared cache...")
                pores_mask = cached_mask > 0
                report(10, f"Pore mask: {np.sum(pores_mask)} voxels")
            elif data.metadata.get("Type", "").startswith("Processed - Void"):
                # Input is already processed pore data
                report(0, "Detected pre-processed pore data...")
                pores_mask = data.raw_data > 0
                report(10, f"Pore mask: {np.sum(pores_mask)} voxels")
            else:
                report(0, f"Starting PNM Extraction (Threshold > {threshold})...")
                
                # Step 1: Segmentation (only if no cache)
                solid_mask = data.raw_data > threshold
                filled_mask = binary_fill_holes(solid_mask)
                pores_mask = filled_mask ^ solid_mask
                
                # Free intermediate arrays
                del solid_mask, filled_mask
                gc.collect()
                
                # Store in shared cache for pore processor to reuse
                porosity = (np.sum(pores_mask) / data.raw_data.size) * 100
                seg_cache.store_pores_mask(volume_id, pores_mask.astype(np.uint8), 
                                          metadata={'porosity': porosity})
            
            report(20, "Segmentation complete. Calculating distance map...")
    
            if np.sum(pores_mask) == 0:
                return self._create_empty_result(data)
    
            # Step 2: Distance Transform (use disk-backed array for large volumes)
            shape = pores_mask.shape
            use_disk = pores_mask.nbytes > 200 * 1024 * 1024  # > 200MB
            
            if use_disk:
                report(25, "Using disk-backed arrays for large volume...")
                cache_dir = tempfile.gettempdir()
                
                # Create disk-backed distance map
                dist_file = os.path.join(cache_dir, f"pnm_dist_{id(data)}.dat")
                distance_map = np.memmap(dist_file, dtype=np.float32, mode='w+', shape=shape)
            else:
                distance_map = np.zeros(shape, dtype=np.float32)
            
            # Compute distance transform in chunks to reduce memory
            report(30, "Computing distance transform...")
            chunk_size = 64
            for i in range(0, shape[0], chunk_size):
                end = min(i + chunk_size, shape[0])
                # Process slice with some overlap for EDT continuity
                start_ext = max(0, i - 10)
                end_ext = min(shape[0], end + 10)
                chunk_dist = distance_transform_edt(pores_mask[start_ext:end_ext])
                # Copy only the non-overlapping part
                offset = i - start_ext
                distance_map[i:end] = chunk_dist[offset:offset + (end - i)]
                del chunk_dist
                gc.collect()
            
            if use_disk:
                distance_map.flush()
            
            report(40, "Distance map computed. Finding local maxima...")
    
            # Find peaks
            local_maxi = find_local_maxima(
                distance_map,
                min_distance=self.MIN_PEAK_DISTANCE,
                labels=pores_mask
            )
            
            # Free pores_mask early
            del pores_mask
            gc.collect()
    
            report(50, f"Found {len(local_maxi)} peaks. Creating markers...")
            
            # Step 3: Watershed segmentation (required for scientific accuracy)
            # Create markers from peaks
            markers = np.zeros(shape, dtype=np.int32)
            if len(local_maxi) > 0:
                markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
            
            report(60, "Running Watershed segmentation (disk-backed)...")
            
            # Always use disk-backed for large volumes to ensure completion
            if use_disk:
                labels_file = os.path.join(cache_dir, f"pnm_labels_{id(data)}.dat")
                segmented_regions = np.memmap(labels_file, dtype=np.int32, mode='w+', shape=shape)
                
                # Process watershed in small chunks to minimize memory
                small_chunk = 32  # Smaller chunks for watershed
                for i in range(0, shape[0], small_chunk):
                    end = min(i + small_chunk, shape[0])
                    chunk_seg = watershed(
                        -distance_map[i:end], 
                        markers[i:end], 
                        mask=distance_map[i:end] > 0
                    )
                    segmented_regions[i:end] = chunk_seg
                    del chunk_seg
                    gc.collect()
                segmented_regions.flush()
            else:
                # In-memory for smaller volumes
                segmented_regions = watershed(-distance_map, markers, mask=distance_map > 0)
            
            num_pores = int(np.max(segmented_regions))
            report(70, f"Watershed complete. Found {num_pores} pores.")
            
            del markers
            gc.collect()
            
            self._cache[cache_key] = {
                'distance_map': distance_map,
                'labels': segmented_regions,
                'num_pores': num_pores
            }

        # Step 3: Mesh Generation
        report(80, "Generating optimized mesh structures (Spheres & Tubes)...")

        pore_centers, pore_radii, pore_ids = self._extract_pore_data(
            segmented_regions, num_pores, data.spacing, data.origin
        )

        if len(pore_centers) > 0:
            pore_cloud = pv.PolyData(pore_centers)
            pore_cloud["radius"] = pore_radii
            pore_cloud["IsPore"] = np.ones(len(pore_centers), dtype=int)
            pore_cloud["ID"] = pore_ids

            sphere_glyph = pv.Sphere(theta_resolution=10, phi_resolution=10)
            pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
            
            n_pts_per_sphere = sphere_glyph.n_points
            pore_radius_scalars = np.repeat(pore_radii, n_pts_per_sphere)
            pores_mesh["PoreRadius"] = pore_radius_scalars
        else:
            pores_mesh = pv.PolyData()

        report(90, "Connecting pores (GPU-accelerated)...")
        
        # GPU-accelerated adjacency detection
        connections = self._find_adjacency_gpu(segmented_regions)
        
        throats_mesh, throat_radii = self._create_throat_mesh_gpu(
            connections, pore_centers, pore_radii, pore_ids,
            distance_map=distance_map, 
            segmented_regions=segmented_regions,
            spacing=data.spacing
        )

        # Step 4: Enhanced Analysis
        report(93, "Calculating advanced metrics...")
        
        size_distribution = self._calculate_size_distribution(pore_radii)
        largest_pore_ratio = self._calculate_connectivity(segmented_regions, num_pores)
        
        # Calculate throat size distribution
        throat_distribution = self._calculate_size_distribution(throat_radii)
        
        throat_stats = {}
        if len(throat_radii) > 0:
            throat_stats = {
                'min': float(np.min(throat_radii)),
                'max': float(np.max(throat_radii)),
                'mean': float(np.mean(throat_radii))
            }
        
        # Scientific Metrics
        total_voxels = segmented_regions.size
        pore_voxels = np.sum(segmented_regions > 0)
        porosity = pore_voxels / total_voxels if total_voxels > 0 else 0
        
        permeability_md = 0.0
        if len(pore_radii) > 0 and porosity > 0 and porosity < 1:
            mean_diameter = 2 * np.mean(pore_radii) * 1e-3
            k_m2 = (porosity**3 * mean_diameter**2) / (180 * (1 - porosity)**2)
            permeability_md = k_m2 / 9.869e-16
        
        tortuosity = 1.0 / np.sqrt(porosity) if porosity > 0 else float('inf')
        
        coordination_number = 0.0
        if num_pores > 0:
            coordination_number = (2 * len(connections)) / num_pores
        
        connected_pore_ids = set()
        for id_a, id_b in connections:
            connected_pore_ids.add(id_a)
            connected_pore_ids.add(id_b)
        connected_pore_fraction = (len(connected_pore_ids) / num_pores * 100) if num_pores > 0 else 0

        # Step 5: Combine
        report(95, "Merging geometry...")
        
        if throats_mesh.n_points > 0:
            throats_mesh["PoreRadius"] = np.zeros(throats_mesh.n_points)
        
        combined_mesh = pores_mesh.merge(throats_mesh)

        report(100, "PNM Generation Complete.")

        return VolumeData(
            raw_data=None,
            mesh=combined_mesh,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - PNM Mesh",
                "PoreCount": int(num_pores),
                "ConnectionCount": len(connections),
                "MeshPoints": combined_mesh.n_points,
                "PoreSizeDistribution": size_distribution,
                "ThroatSizeDistribution": throat_distribution,
                "LargestPoreRatio": f"{largest_pore_ratio:.2f}%",
                "ThroatStats": throat_stats,
                "PoreRadii": pore_radii.tolist(),
                "Porosity": f"{porosity * 100:.2f}%",
                "Permeability_mD": f"{permeability_md:.4f}",
                "Tortuosity": f"{tortuosity:.3f}",
                "CoordinationNumber": f"{coordination_number:.2f}",
                "ConnectedPoreFraction": f"{connected_pore_fraction:.1f}%"
            }
        )

    def _extract_pore_data(self, regions, num_pores, spacing, origin):
        """Extracts centroids and radii for mesh generation."""
        slices = ndimage.find_objects(regions)
        sx, sy, sz = spacing
        ox, oy, oz = origin
        avg_spacing = (sx + sy + sz) / 3.0

        def process_single_pore(i):
            label_idx = i + 1
            slice_obj = slices[i]
            if slice_obj is None:
                return None

            local_mask = (regions[slice_obj] == label_idx)
            voxel_count = np.sum(local_mask)
            if voxel_count == 0:
                return None

            r_vox = (3 * voxel_count / (4 * np.pi)) ** (1 / 3)
            radius = r_vox * avg_spacing

            local_cent = ndimage.center_of_mass(local_mask)

            cz = (slice_obj[0].start + local_cent[0]) * sz + oz
            cy = (slice_obj[1].start + local_cent[1]) * sy + oy
            cx = (slice_obj[2].start + local_cent[2]) * sx + ox

            return (label_idx, [cx, cy, cz], radius)

        # Use parallel processing for large number of pores
        if JOBLIB_AVAILABLE and num_pores > 100:
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(process_single_pore)(i) for i in range(num_pores)
            )
            results = [r for r in results if r is not None]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = []
            max_workers = min(8, max(1, num_pores))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_pore, i): i for i in range(num_pores)}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)

        results.sort(key=lambda x: x[0])
        
        if not results:
            return np.array([]), np.array([]), np.array([])
        
        ids = [r[0] for r in results]
        centers = [r[1] for r in results]
        radii = [r[2] for r in results]

        return np.array(centers), np.array(radii), np.array(ids)

    def _find_adjacency_gpu(self, labels_volume) -> Set[Tuple[int, int]]:
        """
        GPU-accelerated adjacency detection using CuPy.
        Falls back to CPU if GPU unavailable or fails.
        """
        backend = get_gpu_backend()
        
        if GPU_ENABLED and backend.available and CUPY_AVAILABLE:
            try:
                return self._find_adjacency_gpu_impl(labels_volume)
            except Exception as e:
                print(f"[GPU] Adjacency detection failed: {e}, using CPU")
                backend.clear_memory()
        
        return self._find_adjacency_cpu(labels_volume)

    def _find_adjacency_gpu_impl(self, labels_volume) -> Set[Tuple[int, int]]:
        """GPU implementation of adjacency detection."""
        import cupy as cp
        
        backend = get_gpu_backend()
        start = time.time()
        
        labels_gpu = cp.asarray(labels_volume)
        adjacency = set()
        
        # Process each axis
        for axis in range(3):
            if axis == 0:
                curr = labels_gpu[:-1, :, :]
                next_ = labels_gpu[1:, :, :]
            elif axis == 1:
                curr = labels_gpu[:, :-1, :]
                next_ = labels_gpu[:, 1:, :]
            else:
                curr = labels_gpu[:, :, :-1]
                next_ = labels_gpu[:, :, 1:]
            
            # Vectorized mask
            mask = (curr > 0) & (next_ > 0) & (curr != next_)
            
            if cp.any(mask):
                pairs_a = curr[mask]
                pairs_b = next_[mask]
                
                # Transfer pairs to CPU
                pairs_a_cpu = cp.asnumpy(pairs_a)
                pairs_b_cpu = cp.asnumpy(pairs_b)
                
                # Use numpy unique to reduce duplicates before set operations
                stacked = np.stack([np.minimum(pairs_a_cpu, pairs_b_cpu),
                                   np.maximum(pairs_a_cpu, pairs_b_cpu)], axis=1)
                unique_pairs = np.unique(stacked, axis=0)
                
                for a, b in unique_pairs:
                    adjacency.add((int(a), int(b)))
                
                del pairs_a, pairs_b
        
        del labels_gpu
        backend.clear_memory()
        
        elapsed = time.time() - start
        print(f"[GPU] Adjacency detection: {elapsed:.2f}s, found {len(adjacency)} connections")
        
        return adjacency

    def _find_adjacency_cpu(self, labels_volume) -> Set[Tuple[int, int]]:
        """CPU fallback for adjacency detection."""
        start = time.time()
        adjacency = set()
        shape = labels_volume.shape
        chunk_size = 64
        
        def process_axis(axis: int):
            """Process adjacency along a single axis."""
            for i in range(0, shape[axis] - 1, chunk_size):
                end = min(i + chunk_size, shape[axis] - 1)
                slices_curr = [slice(None)] * 3
                slices_next = [slice(None)] * 3
                slices_curr[axis] = slice(i, end)
                slices_next[axis] = slice(i + 1, end + 1)
                
                val_curr = labels_volume[tuple(slices_curr)]
                val_next = labels_volume[tuple(slices_next)]
                mask = (val_curr > 0) & (val_next > 0) & (val_curr != val_next)
                
                if np.any(mask):
                    stacked = np.stack([
                        np.minimum(val_curr[mask], val_next[mask]),
                        np.maximum(val_curr[mask], val_next[mask])
                    ], axis=1)
                    for a, b in np.unique(stacked, axis=0):
                        adjacency.add((int(a), int(b)))
        
        for axis in range(3):
            process_axis(axis)
        
        print(f"[CPU] Adjacency detection: {time.time() - start:.2f}s, found {len(adjacency)} connections")
        return adjacency

    def _create_throat_mesh_gpu(self, connections, centers, radii, ids_list,
                                distance_map=None, segmented_regions=None, spacing=None):
        """
        Creates tubes connecting the pores using GPU-accelerated batch processing.
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
            from concurrent.futures import ThreadPoolExecutor
            
            def measure_throat(conn):
                id_a, id_b = conn
                key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                radius, _ = self._measure_throat_radius_fast(
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
                vertices, faces = self._generate_tubes_batch_gpu(valid_connections, n_sides=6)
                if len(vertices) > 0:
                    throats = pv.PolyData(vertices, faces)
                    throats["IsPore"] = np.zeros(throats.n_points, dtype=int)
                    print(f"[GPU] Generated {len(valid_connections)} tubes")
                    return throats, throat_radii
            except Exception as e:
                print(f"[GPU] Tube generation failed: {e}, using CPU")
                backend.clear_memory()

        # CPU fallback - batch with PyVista
        throats = self._generate_tubes_batch_cpu(valid_connections)
        throats["IsPore"] = np.zeros(throats.n_points, dtype=int)
        
        return throats, throat_radii

    def _generate_tubes_batch_gpu(self, connections: List[Tuple], n_sides: int = 6):
        """Generate all tube meshes on GPU, transfer once at end."""
        import cupy as cp
        
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

    def _generate_tubes_batch_cpu(self, connections: List[Tuple]) -> pv.PolyData:
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

    def _measure_throat_radius_fast(self, distance_map, segmented_regions, 
                                     pore_a: int, pore_b: int, spacing: tuple,
                                     slices_cache: dict = None) -> Tuple[float, Optional[np.ndarray]]:
        """
        OPTIMIZED: Measure throat radius using bounding box to limit computation.
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

    def _calculate_size_distribution(self, pore_radii):
        """Calculate pore size distribution histogram."""
        if len(pore_radii) == 0:
            return {"bins": [], "counts": []}
        
        bins = np.linspace(0, np.max(pore_radii) * 1.1, 10)
        counts, bin_edges = np.histogram(pore_radii, bins=bins)
        
        return {"bins": bin_edges.tolist(), "counts": counts.tolist()}

    def _calculate_connectivity(self, labels_volume, num_pores):
        """Calculate percentage of largest connected pore volume."""
        if num_pores == 0:
            return 0.0
        
        flat_labels = labels_volume.ravel()
        pore_volumes = np.bincount(flat_labels, minlength=num_pores + 1)
        pore_volumes = pore_volumes[1:]
        
        if len(pore_volumes) == 0:
            return 0.0
        
        largest_volume = np.max(pore_volumes)
        total_volume = np.sum(pore_volumes)
        
        return (largest_volume / total_volume) * 100.0 if total_volume > 0 else 0.0

    def _create_empty_result(self, data: VolumeData) -> VolumeData:
        return VolumeData(metadata={"Type": "Empty", "PoreCount": 0})
