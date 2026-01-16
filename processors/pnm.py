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
from config import GPU_ENABLED

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

        pore_centers, pore_radii, pore_ids = self._extract_pore_data(
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

        # Step 2: Distance Transform
        shape = pores_mask.shape
        use_disk = pores_mask.nbytes > 200 * 1024 * 1024  # > 200MB
        
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
        """Extract pore centroids and radii for mesh generation."""
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
