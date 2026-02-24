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
from core.coordinates import voxel_zyx_to_world_xyz
from processors.utils import binary_fill_holes, distance_transform_edt, find_local_maxima
from processors.pore_segmentation import (
    PoreSegmentationConfig,
    segment_pores_from_mask,
    segment_pores_from_raw,
)
from processors.pnm_adjacency import find_adjacency
from processors.pnm_throat import create_throat_mesh
from config import (
    MIN_PEAK_DISTANCE as CFG_MIN_PEAK_DISTANCE,
    SEGMENTATION_PROFILE_DEFAULT,
    SEGMENTATION_SPLIT_MODE_DEFAULT,
)

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

    MIN_PEAK_DISTANCE = CFG_MIN_PEAK_DISTANCE

    def __init__(self):
        super().__init__()
        self._cache = {}

    def process(
        self,
        data: VolumeData,
        callback: Optional[Callable[[int, str], None]] = None,
        threshold: int = -300,
        segmentation_profile: str = SEGMENTATION_PROFILE_DEFAULT,
        split_mode: str = SEGMENTATION_SPLIT_MODE_DEFAULT,
    ) -> VolumeData:
        """
        Process volume data to generate PNM mesh.
        
        If input data is already processed pore data (from PoreExtractionProcessor),
        it will be reused directly, avoiding duplicate segmentation.
        
        Args:
            data: Input VolumeData with raw_data
            callback: Progress callback (percent, message)
            threshold: intensity threshold for solid/void separation
            
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
        volume_id = f"{id(data.raw_data)}_{threshold}_{segmentation_profile}_{split_mode}"
        
        # Internal PNM cache for watershed results
        cache_key = (id(data.raw_data), threshold, segmentation_profile, split_mode)
        cached_result = self._cache.get(cache_key)
        segmentation_debug = {
            "profile": segmentation_profile,
            "split_mode": split_mode,
        }

        if cached_result:
            report(10, "Cache Hit! Reusing watershed results...")
            distance_map = cached_result['distance_map']
            segmented_regions = cached_result['labels']
            num_pores = cached_result['num_pores']
            segmentation_debug = cached_result.get("segmentation_debug", segmentation_debug)
            report(80, "Skipped segmentation. Generating optimized mesh...")
        else:
            # Phase 1: Segmentation
            distance_map, segmented_regions, num_pores, segmentation_debug = self._run_segmentation(
                data,
                threshold,
                volume_id,
                seg_cache,
                report,
                segmentation_profile=segmentation_profile,
                split_mode=split_mode,
            )
            
            if num_pores == 0:
                return self._create_empty_result(data, segmentation_debug=segmentation_debug)
            
            # Cache for future use
            self._cache[cache_key] = {
                'distance_map': distance_map,
                'labels': segmented_regions,
                'num_pores': num_pores,
                'segmentation_debug': segmentation_debug,
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
                "SegmentationDebug": segmentation_debug,
                **metrics
            }
        )

    def _run_segmentation(
        self,
        data,
        threshold,
        volume_id,
        seg_cache,
        report,
        segmentation_profile: str = SEGMENTATION_PROFILE_DEFAULT,
        split_mode: str = SEGMENTATION_SPLIT_MODE_DEFAULT,
    ):
        """Run segmentation pipeline: mask -> distance transform -> watershed."""
        seg_cfg = PoreSegmentationConfig(
            profile=str(segmentation_profile),
            split_mode=str(split_mode),
            base_min_peak_distance=int(self.MIN_PEAK_DISTANCE),
        )

        cached_mask = seg_cache.get_pores_mask(volume_id)
        if cached_mask is not None:
            report(0, "Using cached pores_mask from shared cache...")
            pores_mask = cached_mask > 0
            report(10, f"Pore mask: {np.sum(pores_mask)} voxels")
            seg_result = segment_pores_from_mask(
                pores_mask=pores_mask,
                spacing=data.spacing,
                config=seg_cfg,
            )
            cached_meta = seg_cache.get_metadata(volume_id) or {}
            seg_debug = dict(seg_result.debug)
            seg_debug.update(cached_meta.get("segmentation_debug") or {})
            report(70, f"Segmentation complete. Found {seg_result.num_pores} pores.")
            return seg_result.distance_map, seg_result.labels, seg_result.num_pores, seg_debug

        if data.metadata.get("Type", "").startswith("Processed - Void"):
            report(0, "Detected pre-processed pore data...")
            pores_mask = data.raw_data > 0
            report(10, f"Pore mask: {np.sum(pores_mask)} voxels")
            seg_result = segment_pores_from_mask(
                pores_mask=pores_mask,
                spacing=data.spacing,
                config=seg_cfg,
            )
            seg_debug = dict(seg_result.debug)
            seg_debug.update(data.metadata.get("SegmentationDebug") or {})
            report(70, f"Segmentation complete. Found {seg_result.num_pores} pores.")
            return seg_result.distance_map, seg_result.labels, seg_result.num_pores, seg_debug

        report(0, f"Starting PNM Extraction (Threshold > {threshold}, profile={segmentation_profile}, split={split_mode})...")

        mask_bytes = data.raw_data.size
        size_mb = mask_bytes / (1024 * 1024)
        use_disk = size_mb > 200

        if not use_disk:
            seg_result = segment_pores_from_raw(
                raw=data.raw_data,
                threshold=threshold,
                spacing=data.spacing,
                config=seg_cfg,
            )
            porosity = (np.sum(seg_result.pores_mask) / data.raw_data.size) * 100.0
            seg_cache.store_pores_mask(
                volume_id,
                seg_result.pores_mask.astype(np.uint8),
                metadata={
                    "porosity": porosity,
                    "segmentation_debug": seg_result.debug,
                },
            )
            report(70, f"Segmentation complete. Found {seg_result.num_pores} pores.")
            if seg_result.num_pores == 0:
                return None, None, 0, seg_result.debug
            return seg_result.distance_map, seg_result.labels, seg_result.num_pores, seg_result.debug

        # Legacy disk-backed fallback for very large volumes.
        report(20, "Using disk-backed legacy segmentation for large volume...")
        solid_mask = data.raw_data > threshold
        filled_mask = binary_fill_holes(solid_mask)
        pores_mask = filled_mask ^ solid_mask
        del solid_mask, filled_mask
        gc.collect()

        porosity = (np.sum(pores_mask) / data.raw_data.size) * 100.0
        seg_debug = {
            "profile": "legacy",
            "split_mode": "conservative",
            "legacy_disk_fallback": True,
        }
        seg_cache.store_pores_mask(
            volume_id,
            pores_mask.astype(np.uint8),
            metadata={
                "porosity": porosity,
                "segmentation_debug": seg_debug,
            },
        )

        report(30, "Computing distance transform (disk-backed)...")
        shape = pores_mask.shape
        cache_dir = tempfile.gettempdir()
        dist_file = os.path.join(cache_dir, f"pnm_dist_{id(data)}.dat")
        distance_map = np.memmap(dist_file, dtype=np.float32, mode="w+", shape=shape)

        chunk_size = 64
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            start_ext = max(0, i - 10)
            end_ext = min(shape[0], end + 10)
            chunk_dist = distance_transform_edt(pores_mask[start_ext:end_ext])
            offset = i - start_ext
            distance_map[i:end] = chunk_dist[offset : offset + (end - i)]
            del chunk_dist
            gc.collect()
        distance_map.flush()

        report(40, "Distance map computed. Finding local maxima...")
        local_maxi = find_local_maxima(
            distance_map,
            min_distance=self.MIN_PEAK_DISTANCE,
            labels=pores_mask,
        )
        report(50, f"Found {len(local_maxi)} peaks. Creating markers...")

        markers = np.zeros(shape, dtype=np.int32)
        if len(local_maxi) > 0:
            markers[tuple(local_maxi.T)] = np.arange(len(local_maxi), dtype=np.int32) + 1

        report(60, "Running Watershed segmentation...")
        labels_file = os.path.join(cache_dir, f"pnm_labels_{id(data)}.dat")
        segmented_regions = np.memmap(labels_file, dtype=np.int32, mode="w+", shape=shape)
        small_chunk = 32
        for i in range(0, shape[0], small_chunk):
            end = min(i + small_chunk, shape[0])
            chunk_seg = watershed(
                -distance_map[i:end],
                markers[i:end],
                mask=distance_map[i:end] > 0,
            )
            segmented_regions[i:end] = chunk_seg
            del chunk_seg
            gc.collect()
        segmented_regions.flush()

        num_pores = int(np.max(segmented_regions)) if segmented_regions.size else 0
        seg_debug.update(
            {
                "num_components": int(ndimage.label(pores_mask, structure=np.ones((3, 3, 3), dtype=bool))[1]),
                "num_forced_seeds": 0,
                "num_neck_split_seeds": 0,
                "num_markers": int(len(local_maxi)),
                "num_pores": int(num_pores),
            }
        )
        report(70, f"Watershed complete. Found {num_pores} pores.")

        del markers, pores_mask
        gc.collect()
        return distance_map, segmented_regions, num_pores, seg_debug

    def _extract_pore_data(self, regions, num_pores, spacing, origin):
        """Extract pore centroids, radii, and volumes for mesh generation and tracking."""
        slices = ndimage.find_objects(regions)
        sx, sy, sz = spacing
        voxel_volume = float(sx) * float(sy) * float(sz)

        def process_single_pore(i):
            label_idx = i + 1
            slice_obj = slices[i]
            if slice_obj is None:
                return None

            local_mask = (regions[slice_obj] == label_idx)
            voxel_count = np.sum(local_mask)
            if voxel_count == 0:
                return None

            # Equivalent-sphere radius in world units from physical pore volume.
            physical_volume = float(voxel_count) * voxel_volume
            radius = (3.0 * physical_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

            local_cent = ndimage.center_of_mass(local_mask)
            center_z = slice_obj[0].start + local_cent[0]
            center_y = slice_obj[1].start + local_cent[1]
            center_x = slice_obj[2].start + local_cent[2]
            cx, cy, cz = voxel_zyx_to_world_xyz(center_z, center_y, center_x, spacing, origin)

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

    def extract_snapshot(
        self,
        data: VolumeData,
        callback: Optional[Callable[[int, str], None]] = None,
        threshold: int = -300,
        time_index: int = 0,
        compute_connectivity: bool = True,
        segmentation_profile: str = SEGMENTATION_PROFILE_DEFAULT,
        split_mode: str = SEGMENTATION_SPLIT_MODE_DEFAULT,
    ) -> 'PNMSnapshot':
        """
        Extract a PNMSnapshot for 4D CT tracking.
        
        This performs the same segmentation as process() but returns
        a PNMSnapshot data structure suitable for temporal tracking
        instead of a mesh.
        
        Args:
            data: Input VolumeData with raw_data
            callback: Progress callback (percent, message)
            threshold: intensity threshold for solid/void separation
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
        volume_id = f"{id(data.raw_data)}_{threshold}_{segmentation_profile}_{split_mode}_snapshot"
        
        # Run segmentation
        distance_map, segmented_regions, num_pores, segmentation_debug = self._run_segmentation(
            data,
            threshold,
            volume_id,
            seg_cache,
            report,
            segmentation_profile=segmentation_profile,
            split_mode=split_mode,
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
                metadata={
                    "threshold": threshold,
                    "num_pores": 0,
                    "SegmentationDebug": segmentation_debug,
                },
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
                'num_connections': len(connections),
                'SegmentationDebug': segmentation_debug,
            }
        )

    def _create_pores_mesh(self, centers, radii, ids, compression_ratios=None):
        """Create sphere glyphs for pore visualization."""
        if len(centers) == 0:
            return pv.PolyData()
            
        pore_cloud = pv.PolyData(centers)
        pore_cloud["radius"] = radii
        pore_cloud["IsPore"] = np.ones(len(centers), dtype=int)
        pore_cloud["ID"] = ids
        if compression_ratios is not None:
            pore_cloud["CompressionRatio"] = compression_ratios

        # Use a unit-radius source so glyph scalar maps 1:1 to physical radius.
        sphere_glyph = pv.Sphere(radius=0.8, theta_resolution=10, phi_resolution=10)
        pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
        
        n_pts_per_sphere = sphere_glyph.n_points
        pore_radius_scalars = np.repeat(radii, n_pts_per_sphere)
        pores_mesh["PoreRadius"] = pore_radius_scalars
        
        # Repeat ID scalars for each sphere vertex (needed for highlighting)
        pore_id_scalars = np.repeat(ids, n_pts_per_sphere)
        pores_mesh["ID"] = pore_id_scalars

        if compression_ratios is not None:
            comp_scalars = np.repeat(compression_ratios, n_pts_per_sphere)
            pores_mesh["CompressionRatio"] = comp_scalars
        else:
            pores_mesh["CompressionRatio"] = np.ones(len(pores_mesh.points))
        
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

    def _create_empty_result(self, data: VolumeData, segmentation_debug: Optional[dict] = None) -> VolumeData:
        metadata = {"Type": "Empty", "PoreCount": 0}
        if segmentation_debug is not None:
            metadata["SegmentationDebug"] = segmentation_debug
        return VolumeData(metadata=metadata)

    def create_time_varying_mesh(self,
                                 reference_mesh: VolumeData,
                                 reference_snapshot,
                                 tracking_result,
                                 timepoint: int,
                                 current_snapshot=None) -> VolumeData:
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
        compression_ratios = np.ones(len(ref_ids), dtype=np.float64)
        
        # Get ID mapping for this timepoint if available
        id_mapping = tracking_result.id_mapping.get(timepoint, {})
        
        # Index current centers by ID for fast lookup if snapshot available
        current_center_map = {}
        if current_snapshot:
            current_ids = current_snapshot.pore_ids
            current_pos = current_snapshot.pore_centers
            for i, pid in enumerate(current_ids):
                current_center_map[int(pid)] = current_pos[i]

        center_history = getattr(tracking_result, "center_history", None)
        
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
                compression_ratios[i] = vol_ratio
            elif current_vol <= 0:
                # Compressed pore: very small but visible
                new_radii[i] = ref_radii[i] * 0.1
                compression_ratios[i] = 0.0
                # Mark as compressed (we might want to hide it, but user asked for "compressed" view)
            else:
                new_radii[i] = ref_radii[i]
                compression_ratios[i] = 1.0
                
    
            # 2. Update Position
            # Prefer stabilized center history if available
            if center_history and pore_id in center_history and timepoint < len(center_history[pore_id]):
                new_centers[i] = np.array(center_history[pore_id][timepoint], dtype=np.float64)
            elif current_snapshot:
                matched_id = id_mapping.get(pore_id, -1)
                if matched_id != -1 and matched_id in current_center_map:
                    new_centers[i] = current_center_map[matched_id]
        
        # Create new sphere mesh with updated radii AND positions

        pores_mesh = self._create_pores_mesh(new_centers, new_radii, ref_ids, compression_ratios=compression_ratios)
        
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
        if throats_mesh.n_points > 0:
            throats_mesh["CompressionRatio"] = np.ones(throats_mesh.n_points, dtype=np.float64)
            throats_mesh["ID"] = np.zeros(throats_mesh.n_points, dtype=ref_ids.dtype)
        
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

