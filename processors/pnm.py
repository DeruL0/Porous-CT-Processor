"""
Pore to sphere processor for Pore Network Modeling (PNM).
GPU-accelerated with CuPy, memory-optimized with disk caching for large volumes.

This module orchestrates the PNM workflow:
1. Segmentation (with shared cache)
2. Distance transform & watershed  
3. Pore extraction
4. Adjacency detection (delegated to adjacency.py)
5. Throat mesh generation (delegated to throat.py)
"""

import os
import tempfile
import numpy as np
import scipy.ndimage as ndimage
import pyvista as pv
from skimage.segmentation import watershed
from typing import Optional, Callable
import gc

from core import BaseProcessor, VolumeData
from processors.utils import binary_fill_holes, distance_transform_edt, find_local_maxima
from processors.pnm_adjacency import find_adjacency
from processors.pnm_throat import create_throat_mesh
from config import GPU_ENABLED, GPU_MIN_SIZE_MB
from core.gpu_backend import CUPY_AVAILABLE, get_gpu_backend

# Optional high-performance libraries
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class PoreToSphereProcessor(BaseProcessor):
    """
    GPU-Optimized Pore Network Modeling (PNM).
    
    Generates a mesh representation of pore space with:
    - Spheres representing pores (sized by equivalent radius)
    - Tubes representing throats (connections between adjacent pores)
    
    Workflow:
    1. Create binary pore mask (with shared cache)
    2. Distance transform for pore sizing
    3. Watershed segmentation for individual pores
    4. Extract pore centers and radii
    5. Detect adjacency between pores
    6. Generate mesh geometry
    """

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
        
        Args:
            data: Input VolumeData with raw_data
            callback: Progress callback (percent, message)
            threshold: HU threshold for solid/void separation
            
        Returns:
            VolumeData with mesh attribute containing PNM geometry
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
            # Phase 1: Segmentation
            distance_map, segmented_regions, num_pores = self._run_segmentation(
                data, threshold, volume_id, seg_cache, report
            )
            
            if num_pores == 0:
                return self._create_empty_result(data)
            
            # Cache for future use
            self._cache[cache_key] = {
                'distance_map': distance_map,
                'labels': segmented_regions,
                'num_pores': num_pores
            }

        # Phase 2: Mesh Generation
        report(80, "Generating optimized mesh structures (Spheres & Tubes)...")

        pore_centers, pore_radii, pore_ids, _ = self._extract_pore_data(
            segmented_regions, num_pores, data.spacing, data.origin
        )

        pores_mesh = self._create_pores_mesh(pore_centers, pore_radii, pore_ids)

        report(90, "Connecting pores (GPU-accelerated)...")
        
        # GPU-accelerated adjacency detection (delegated)
        connections = find_adjacency(segmented_regions)
        
        # Throat mesh generation (delegated)
        throats_mesh, throat_radii = create_throat_mesh(
            connections, pore_centers, pore_radii, pore_ids,
            distance_map=distance_map, 
            segmented_regions=segmented_regions,
            spacing=data.spacing
        )

        # Phase 3: Metrics & Assembly
        report(93, "Calculating advanced metrics...")
        
        metrics = self._calculate_metrics(
            pore_radii, throat_radii, segmented_regions, num_pores, connections
        )

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
                **metrics
            }
        )

    def _run_segmentation(self, data, threshold, volume_id, seg_cache, report):
        """Run segmentation pipeline: mask -> distance transform -> watershed."""
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
            
            # Step 1: Segmentation
            solid_mask = data.raw_data > threshold
            filled_mask = binary_fill_holes(solid_mask)
            pores_mask = filled_mask ^ solid_mask
            
            del solid_mask, filled_mask
            gc.collect()
            
            # Store in shared cache
            porosity = (np.sum(pores_mask) / data.raw_data.size) * 100
            seg_cache.store_pores_mask(volume_id, pores_mask.astype(np.uint8), 
                                      metadata={'porosity': porosity})
        
        report(20, "Segmentation complete. Calculating distance map...")

        if np.sum(pores_mask) == 0:
            return None, None, 0

        # Check if we can use unified GPU pipeline (keeps data on GPU)
        shape = pores_mask.shape
        size_mb = pores_mask.nbytes / (1024 * 1024)
        use_disk = size_mb > 200  # > 200MB
        
        if GPU_ENABLED and CUPY_AVAILABLE and not use_disk:
            backend = get_gpu_backend()
            free_mb = backend.get_free_memory_mb()
            required_mb = size_mb * 6  # Pipeline needs ~6x memory
            
            if required_mb < free_mb * 0.8:
                report(30, "Using unified GPU pipeline...")
                try:
                    from processors.gpu_pipeline import run_segmentation_pipeline_gpu
                    distance_map, segmented_regions, num_pores = run_segmentation_pipeline_gpu(
                        pores_mask, min_peak_distance=self.MIN_PEAK_DISTANCE
                    )
                    del pores_mask
                    gc.collect()
                    report(70, f"GPU pipeline complete. Found {num_pores} pores.")
                    return distance_map, segmented_regions, num_pores
                except Exception as e:
                    report(30, f"GPU pipeline failed: {e}, falling back to chunked...")
        
        # Fallback: chunked processing for large volumes or when GPU unavailable
        if use_disk:
            report(25, "Using disk-backed arrays for large volume...")
            cache_dir = tempfile.gettempdir()
            dist_file = os.path.join(cache_dir, f"pnm_dist_{id(data)}.dat")
            distance_map = np.memmap(dist_file, dtype=np.float32, mode='w+', shape=shape)
        else:
            distance_map = np.zeros(shape, dtype=np.float32)
        
        # Compute distance transform in chunks
        report(30, "Computing distance transform...")
        chunk_size = 64
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            start_ext = max(0, i - 10)
            end_ext = min(shape[0], end + 10)
            chunk_dist = distance_transform_edt(pores_mask[start_ext:end_ext])
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
        
        del pores_mask
        gc.collect()

        report(50, f"Found {len(local_maxi)} peaks. Creating markers...")
        
        # Step 3: Watershed segmentation
        markers = np.zeros(shape, dtype=np.int32)
        if len(local_maxi) > 0:
            markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
        
        report(60, "Running Watershed segmentation...")
        
        if use_disk:
            cache_dir = tempfile.gettempdir()
            labels_file = os.path.join(cache_dir, f"pnm_labels_{id(data)}.dat")
            segmented_regions = np.memmap(labels_file, dtype=np.int32, mode='w+', shape=shape)
            
            small_chunk = 32
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
            segmented_regions = watershed(-distance_map, markers, mask=distance_map > 0)
        
        num_pores = int(np.max(segmented_regions))
        report(70, f"Watershed complete. Found {num_pores} pores.")
        
        del markers
        gc.collect()
        
        return distance_map, segmented_regions, num_pores

    def _extract_pore_data(self, regions, num_pores, spacing, origin):
        """Extract pore centroids, radii, and volumes for mesh generation and tracking."""
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

            return (label_idx, [cx, cy, cz], radius, int(voxel_count))

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
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        ids = [r[0] for r in results]
        centers = [r[1] for r in results]
        radii = [r[2] for r in results]
        volumes = [r[3] for r in results]

        return np.array(centers), np.array(radii), np.array(ids), np.array(volumes)

    def extract_snapshot(self, data: VolumeData, 
                        callback: Optional[Callable[[int, str], None]] = None,
                        threshold: int = -300,
                        time_index: int = 0,
                        compute_connectivity: bool = True) -> 'PNMSnapshot':
        """
        Extract a PNMSnapshot for 4D CT tracking.
        
        This performs the same segmentation as process() but returns
        a PNMSnapshot data structure suitable for temporal tracking
        instead of a mesh.
        
        Args:
            data: Input VolumeData with raw_data
            callback: Progress callback (percent, message)
            threshold: HU threshold for solid/void separation
            time_index: Time index for this snapshot
            compute_connectivity: If False, skip connection computation (saves time for t>0)
            
        Returns:
            PNMSnapshot with pore data and segmented regions
        """
        from core.time_series import PNMSnapshot
        
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        def report(p, msg):
            print(f"[PNM Snapshot] {msg}")
            if callback: callback(p, msg)

        from data.disk_cache import get_segmentation_cache
        
        seg_cache = get_segmentation_cache()
        volume_id = f"{id(data.raw_data)}_{threshold}_snapshot"
        
        # Run segmentation
        distance_map, segmented_regions, num_pores = self._run_segmentation(
            data, threshold, volume_id, seg_cache, report
        )
        
        if num_pores == 0:
            return PNMSnapshot(
                time_index=time_index,
                pore_centers=np.array([]),
                pore_radii=np.array([]),
                pore_ids=np.array([]),
                pore_volumes=np.array([]),
                connections=[],
                segmented_regions=segmented_regions,
                spacing=data.spacing,
                origin=data.origin,
                metadata={'threshold': threshold, 'num_pores': 0}
            )
        
        # Extract pore data
        report(80, "Extracting pore data for tracking...")
        pore_centers, pore_radii, pore_ids, pore_volumes = self._extract_pore_data(
            segmented_regions, num_pores, data.spacing, data.origin
        )
        
        # Find connections only for reference frame (t=0)
        # For subsequent frames, connectivity is inherited from reference
        if compute_connectivity:
            report(90, "Finding pore connections...")
            connections = find_adjacency(segmented_regions)
            report(100, f"Snapshot extracted: {len(pore_ids)} pores, {len(connections)} connections")
        else:
            connections = []  # Will be filled from reference snapshot
            report(100, f"Snapshot extracted: {len(pore_ids)} pores (connectivity from reference)")
        
        return PNMSnapshot(
            time_index=time_index,
            pore_centers=pore_centers,
            pore_radii=pore_radii,
            pore_ids=pore_ids,
            pore_volumes=pore_volumes,
            connections=connections,
            segmented_regions=segmented_regions,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                'threshold': threshold,
                'num_pores': len(pore_ids),
                'num_connections': len(connections)
            }
        )

    def _create_pores_mesh(self, centers, radii, ids):
        """Create sphere glyphs for pore visualization."""
        if len(centers) == 0:
            return pv.PolyData()
            
        pore_cloud = pv.PolyData(centers)
        pore_cloud["radius"] = radii
        pore_cloud["IsPore"] = np.ones(len(centers), dtype=int)
        pore_cloud["ID"] = ids

        sphere_glyph = pv.Sphere(theta_resolution=10, phi_resolution=10)
        pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
        
        n_pts_per_sphere = sphere_glyph.n_points
        pore_radius_scalars = np.repeat(radii, n_pts_per_sphere)
        pores_mesh["PoreRadius"] = pore_radius_scalars
        
        return pores_mesh

    def _calculate_metrics(self, pore_radii, throat_radii, segmented_regions, num_pores, connections):
        """Calculate scientific metrics for pore network analysis."""
        # Size distributions
        size_distribution = self._calculate_size_distribution(pore_radii)
        throat_distribution = self._calculate_size_distribution(throat_radii)
        
        # Throat statistics
        throat_stats = {}
        if len(throat_radii) > 0:
            throat_stats = {
                'min': float(np.min(throat_radii)),
                'max': float(np.max(throat_radii)),
                'mean': float(np.mean(throat_radii))
            }
        
        # Porosity
        total_voxels = segmented_regions.size
        pore_voxels = np.sum(segmented_regions > 0)
        porosity = pore_voxels / total_voxels if total_voxels > 0 else 0
        
        # Permeability (Kozeny-Carman)
        permeability_md = 0.0
        if len(pore_radii) > 0 and porosity > 0 and porosity < 1:
            mean_diameter = 2 * np.mean(pore_radii) * 1e-3
            k_m2 = (porosity**3 * mean_diameter**2) / (180 * (1 - porosity)**2)
            permeability_md = k_m2 / 9.869e-16
        
        # Tortuosity
        tortuosity = 1.0 / np.sqrt(porosity) if porosity > 0 else float('inf')
        
        # Coordination number
        coordination_number = 0.0
        if num_pores > 0:
            coordination_number = (2 * len(connections)) / num_pores
        
        # Connected pore fraction
        connected_pore_ids = set()
        for id_a, id_b in connections:
            connected_pore_ids.add(id_a)
            connected_pore_ids.add(id_b)
        connected_pore_fraction = (len(connected_pore_ids) / num_pores * 100) if num_pores > 0 else 0
        
        # Largest pore ratio
        largest_pore_ratio = self._calculate_connectivity(segmented_regions, num_pores)

        return {
            "PoreSizeDistribution": size_distribution,
            "ThroatSizeDistribution": throat_distribution,
            "LargestPoreRatio": f"{largest_pore_ratio:.2f}%",
            "ThroatStats": throat_stats,
            "PoreRadii": pore_radii.tolist() if len(pore_radii) > 0 else [],
            "Porosity": f"{porosity * 100:.2f}%",
            "Permeability_mD": f"{permeability_md:.4f}",
            "Tortuosity": f"{tortuosity:.3f}",
            "CoordinationNumber": f"{coordination_number:.2f}",
            "ConnectedPoreFraction": f"{connected_pore_fraction:.1f}%"
        }

    def _calculate_size_distribution(self, radii):
        """Calculate size distribution histogram."""
        if len(radii) == 0:
            return {"bins": [], "counts": []}
        
        bins = np.linspace(0, np.max(radii) * 1.1, 10)
        counts, bin_edges = np.histogram(radii, bins=bins)
        
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

    def create_time_varying_mesh(self, 
                               reference_mesh: VolumeData,

                                  reference_snapshot,
                                  tracking_result,
                                  timepoint: int,
                                  current_snapshot = None) -> VolumeData:
        """
        Create a PNM mesh for a specific timepoint using the reference structure.
        
        The topology (connectivity) remains the same as the reference.
        Sphere radii and POSITIONS are updated based on tracking data.
        
        Args:
            reference_mesh: The VolumeData with mesh from t=0
            reference_snapshot: PNMSnapshot from t=0 with pore data
            tracking_result: PoreTrackingResult with volume history
            timepoint: Which timepoint to generate mesh for
            current_snapshot: Optional PNMSnapshot for the target timepoint (for positions)
            
        Returns:
            VolumeData with mesh adjusted for the given timepoint
        """
        if not reference_mesh.has_mesh:
            raise ValueError("Reference mesh must contain a mesh")
        
        # Get reference data
        ref_centers = reference_snapshot.pore_centers
        ref_radii = reference_snapshot.pore_radii
        ref_ids = reference_snapshot.pore_ids
        ref_volumes = reference_snapshot.pore_volumes
        
        # Calculate new radii and positions
        new_radii = np.zeros_like(ref_radii, dtype=np.float64)
        new_centers = np.copy(ref_centers) # Default to reference positions
        
        # Get ID mapping for this timepoint if available
        id_mapping = tracking_result.id_mapping.get(timepoint, {})
        
        # Index current centers by ID for fast lookup if snapshot available
        current_center_map = {}
        if current_snapshot:
            current_ids = current_snapshot.pore_ids
            current_pos = current_snapshot.pore_centers
            for i, pid in enumerate(current_ids):
                current_center_map[int(pid)] = current_pos[i]
        
        for i, pore_id in enumerate(ref_ids):
            pore_id = int(pore_id)
            ref_vol = ref_volumes[i] if i < len(ref_volumes) else 1.0
            
            # 1. Update Volume/Radius
            vol_history = tracking_result.volume_history.get(pore_id, [])
            if timepoint < len(vol_history):
                current_vol = vol_history[timepoint]
            else:
                current_vol = ref_vol
            
            # Scale radius by cube root of volume ratio (volume ~ r³)
            if ref_vol > 0 and current_vol > 0:
                vol_ratio = current_vol / ref_vol
                new_radii[i] = ref_radii[i] * (vol_ratio ** (1/3))
            elif current_vol <= 0:
                # Compressed pore: very small but visible
                new_radii[i] = ref_radii[i] * 0.1
                # Mark as compressed (we might want to hide it, but user asked for "compressed" view)
            else:
                new_radii[i] = ref_radii[i]
                
    
            # 2. Update Position
            # We need to know which ID in the current snapshot corresponds to this reference pore_id
            if current_snapshot:
                # Get the matched ID in current frame
                matched_id = id_mapping.get(pore_id, -1)
                
                if matched_id != -1 and matched_id in current_center_map:
                    # Move sphere to the new tracked position
                    new_centers[i] = current_center_map[matched_id]
        
        # Create new sphere mesh with updated radii AND positions

        pores_mesh = self._create_pores_mesh(new_centers, new_radii, ref_ids)
        
        # Regenerate Throat Mesh using new centers (connectivity remains unchanged)
        # reference_snapshot.connections stores (id_a, id_b)
        connections = reference_snapshot.connections
        
        # Create new throat mesh
        # We don't use distance_map or segmented_regions here to keep it fast and 
        # because the connectivity is strictly from the reference frame.
        throats_mesh, throat_radii = create_throat_mesh(
            connections, new_centers, new_radii, ref_ids,
            distance_map=None, segmented_regions=None, spacing=reference_mesh.spacing
        )
        
        # Combine
        combined_mesh = pores_mesh.merge(throats_mesh)
        
        # Calculate stats for this timepoint
        active_count = sum(1 for pid in ref_ids 
                         if pid in tracking_result.status_history and
                         timepoint < len(tracking_result.status_history[pid]) and
                         tracking_result.status_history[pid][timepoint].value == 'active')
        
        # Calculate distributions for dynamic histograms
        pore_dist = self._calculate_size_distribution(new_radii)
        throat_dist = self._calculate_size_distribution(throat_radii)
        
        # Calculate throat stats
        throat_stats = {}
        if len(throat_radii) > 0:
            throat_stats = {
                'min': float(np.min(throat_radii)),
                'max': float(np.max(throat_radii)),
                'mean': float(np.mean(throat_radii))
            }

        return VolumeData(
            raw_data=None,
            mesh=combined_mesh,
            spacing=reference_mesh.spacing,
            origin=reference_mesh.origin,
            metadata={
                "Type": f"Processed - PNM Mesh (t={timepoint})",
                "PoreCount": len(ref_ids),
                "ActivePores": active_count,
                "CompressedPores": len(ref_ids) - active_count,
                "Timepoint": timepoint,
                "ConnectionCount": len(connections),
                "PoreSizeDistribution": pore_dist,
                "ThroatSizeDistribution": throat_dist,
                "ThroatStats": throat_stats,
                "LargestPoreRatio": reference_mesh.metadata.get("LargestPoreRatio", "N/A")
            }
        )

