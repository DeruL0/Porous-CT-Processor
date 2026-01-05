import numpy as np
import scipy.ndimage as ndimage
import pyvista as pv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Tuple, Dict, Set, Optional, Callable
from Core import BaseProcessor, VolumeData


# ==========================================
# Image Processors
# ==========================================

class PoreExtractionProcessor(BaseProcessor):
    """
    Segments the void space (pores) from the solid matrix.
    Returns a Voxel-based VolumeData object representing "Air".
    """

    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None,
                threshold: int = -300) -> VolumeData:
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        def report(p: int, msg: str):
            print(f"[Processor] {msg}")
            if callback: callback(p, msg)

        report(0, f"Starting pore detection (Threshold < {threshold})...")

        # 1. Binarization (Air vs Solid)
        solid_mask = data.raw_data > threshold
        report(20, "Binarization complete. Filling holes...")

        # 2. Morphology
        filled_volume = ndimage.binary_fill_holes(solid_mask)
        pores_mask = filled_volume ^ solid_mask
        report(50, "Morphology operations complete. Calculating stats...")

        # Quantitative Analysis
        pore_voxels = np.sum(pores_mask)
        total_voxels = data.raw_data.size
        porosity_pct = (pore_voxels / total_voxels) * 100.0

        # Label connected components
        report(70, "Labeling connected components (this may take a while)...")
        labeled_array, num_features = ndimage.label(pores_mask, structure=np.ones((3, 3, 3)))

        report(90, f"Found {num_features} pores. Generating output volume...")

        # Create Output Volume (Voxel Grid)
        processed_volume = np.zeros_like(data.raw_data)
        processed_volume[pores_mask] = 1000  # Highlight Pores

        report(100, "Processing complete.")

        return VolumeData(
            raw_data=processed_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Void Volume",
                "Porosity": f"{porosity_pct:.2f}%",
                "PoreCount": int(num_features)
            }
        )


class PoreToSphereProcessor(BaseProcessor):
    """
    Optimized Pore Network Modeling (PNM).
    """

    MIN_PEAK_DISTANCE = 6

    def __init__(self):
        super().__init__()
        # Simple in-memory cache
        # Key: (data_id, threshold)
        # Value: { 'distance_map': ..., 'labels': ..., 'num_pores': ... }
        self._cache = {}

    def process(self, data: VolumeData, callback: Optional[Callable[[int, str], None]] = None,
                threshold: int = -300) -> VolumeData:
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        def report(p, msg):
            print(f"[PNM] {msg}")
            if callback: callback(p, msg)

        # Generate a semi-unique ID for the input data to allow caching
        # We use memory address + shape + sum (cheap-ish hash) or just object ID if we assume immutability logic
        # For safety in this demo, we'll use object ID + threshold
        cache_key = (id(data.raw_data), threshold)
        cached_result = self._cache.get(cache_key)

        if cached_result:
            report(10, "Cache Hit! Reusing segmentation results...")
            distance_map = cached_result['distance_map']
            segmented_regions = cached_result['labels']
            num_pores = cached_result['num_pores']
            
            # Skip to mesh generation
            report(80, "Skipped segmentation. Generating optimized mesh...")
        else:
            report(0, f"Starting PNM Extraction (Threshold > {threshold})...")
            
            # --- Step 1: Segmentation ---
            solid_mask = data.raw_data > threshold
            filled_mask = ndimage.binary_fill_holes(solid_mask)
            pores_mask = filled_mask ^ solid_mask
    
            report(20, "Segmentation complete. Calculating distance map...")
    
            if np.sum(pores_mask) == 0:
                return self._create_empty_result(data)
    
            # --- Step 2: Watershed Analysis ---
            distance_map = ndimage.distance_transform_edt(pores_mask)
            report(40, "Distance map computed. Finding local maxima...")
    
            local_maxi = peak_local_max(
                distance_map,
                labels=pores_mask,
                min_distance=self.MIN_PEAK_DISTANCE
            )
    
            markers = np.zeros_like(distance_map, dtype=int)
            markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1
    
            report(60, "Running Watershed segmentation...")
            segmented_regions = watershed(-distance_map, markers, mask=pores_mask)
            num_pores = np.max(segmented_regions)
            
            # Save to Cache
            self._cache[cache_key] = {
                'distance_map': distance_map,
                'labels': segmented_regions,
                'num_pores': num_pores
            }

        # --- Step 3: Mesh Generation ---
        report(80, "Generating optimized mesh structures (Spheres & Tubes)...")

        # A. Nodes (Pores) -> Spheres
        pore_centers, pore_radii, pore_ids = self._extract_pore_data(segmented_regions, num_pores, data.spacing,
                                                                     data.origin)

        # Create PyVista PolyData for Pores
        if len(pore_centers) > 0:
            pore_cloud = pv.PolyData(pore_centers)
            pore_cloud["radius"] = pore_radii
            pore_cloud["IsPore"] = np.ones(len(pore_centers), dtype=int)
            pore_cloud["ID"] = pore_ids

            sphere_glyph = pv.Sphere(theta_resolution=10, phi_resolution=10)
            pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
        else:
            pores_mesh = pv.PolyData()

        # B. Edges (Throats) -> Tubes
        report(90, "Connecting pores...")
        connections = self._find_adjacency(segmented_regions)
        throats_mesh, throat_radii = self._create_throat_mesh(connections, pore_centers, pore_radii, pore_ids)

        # --- Step 4: Enhanced Analysis ---
        report(93, "Calculating advanced metrics...")
        
        # Pore size distribution (histogram bins)
        size_distribution = self._calculate_size_distribution(pore_radii)
        
        # Connectivity analysis
        largest_pore_ratio = self._calculate_connectivity(segmented_regions, num_pores)
        
        # Throat statistics
        throat_stats = {}
        if len(throat_radii) > 0:
            throat_stats = {
                'min': float(np.min(throat_radii)),
                'max': float(np.max(throat_radii)),
                'mean': float(np.mean(throat_radii))
            }

        # --- Step 5: Combine ---
        report(95, "Merging geometry...")
        combined_mesh = pores_mesh.merge(throats_mesh)

        report(100, "PNM Generation Complete.")

        return VolumeData(
            raw_data=None,  # No Voxel data in this result, purely Mesh
            mesh=combined_mesh,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - PNM Mesh",
                "PoreCount": int(num_pores),
                "ConnectionCount": len(connections),
                "MeshPoints": combined_mesh.n_points,
                # Enhanced metrics
                "PoreSizeDistribution": size_distribution,
                "LargestPoreRatio": f"{largest_pore_ratio:.2f}%",
                "ThroatStats": throat_stats,
                "PoreRadii": pore_radii.tolist()  # For visualization
            }
        )

    def _extract_pore_data(self, regions, num_pores, spacing, origin):
        """Extracts centroids and radii for mesh generation."""
        slices = ndimage.find_objects(regions)
        centers = []
        radii = []
        ids = []

        sx, sy, sz = spacing
        ox, oy, oz = origin

        for i in range(num_pores):
            label_idx = i + 1
            slice_obj = slices[i]
            if slice_obj is None: continue

            local_mask = (regions[slice_obj] == label_idx)
            voxel_count = np.sum(local_mask)

            # Equivalent radius
            r_vox = (3 * voxel_count / (4 * np.pi)) ** (1 / 3)
            # Take average spacing for radius scaling
            avg_spacing = (sx + sy + sz) / 3.0
            radii.append(r_vox * avg_spacing)

            # Centroid (Physical Coordinates)
            local_cent = ndimage.center_of_mass(local_mask)

            cz = (slice_obj[0].start + local_cent[0]) * sz + oz
            cy = (slice_obj[1].start + local_cent[1]) * sy + oy
            cx = (slice_obj[2].start + local_cent[2]) * sx + ox

            # PyVista uses (X, Y, Z) order, Numpy uses (Z, Y, X)
            centers.append([cx, cy, cz])
            ids.append(label_idx)

        return np.array(centers), np.array(radii), np.array(ids)

    def _find_adjacency(self, labels_volume) -> Set[Tuple[int, int]]:
        """Identifies connected pores."""
        adjacency = set()
        shifts = [
            (np.s_[:-1, :, :], np.s_[1:, :, :]),
            (np.s_[:, :-1, :], np.s_[:, 1:, :]),
            (np.s_[:, :, :-1], np.s_[:, :, 1:])
        ]

        for s_curr, s_next in shifts:
            val_curr = labels_volume[s_curr]
            val_next = labels_volume[s_next]
            mask = (val_curr > 0) & (val_next > 0) & (val_curr != val_next)

            if np.any(mask):
                pairs = np.stack((val_curr[mask], val_next[mask]), axis=-1)
                unique_pairs = np.unique(np.sort(pairs, axis=1), axis=0)
                for p in unique_pairs:
                    adjacency.add(tuple(p))
        return adjacency

    def _create_throat_mesh(self, connections, centers, radii, ids_list):
        """Creates tubes connecting the pores. Returns (mesh, throat_radii_list)."""
        if not connections:
            return pv.PolyData(), np.array([])

        # Map ID to Index in 'centers' array
        id_map = {uid: idx for idx, uid in enumerate(ids_list)}

        lines = []
        throat_radii = []

        for id_a, id_b in connections:
            if id_a in id_map and id_b in id_map:
                idx_a = id_map[id_a]
                idx_b = id_map[id_b]

                p1 = centers[idx_a]
                p2 = centers[idx_b]

                # Estimate throat radius (smaller than pores)
                r_throat = min(radii[idx_a], radii[idx_b]) * 0.3
                throat_radii.append(r_throat)

                # Create a single tube
                line = pv.Line(p1, p2)
                tube = line.tube(radius=r_throat)
                lines.append(tube)

        if not lines:
            return pv.PolyData(), np.array([])

        # Merge all tubes
        throats = lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]
        throats["IsPore"] = np.zeros(throats.n_points, dtype=int)

        return throats, np.array(throat_radii)

    def _calculate_size_distribution(self, pore_radii):
        """Calculate pore size distribution histogram."""
        if len(pore_radii) == 0:
            return {"bins": [], "counts": []}
        
        # Create bins (adjust based on your data range)
        bins = np.linspace(0, np.max(pore_radii) * 1.1, 10)
        counts, bin_edges = np.histogram(pore_radii, bins=bins)
        
        return {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist()
        }

    def _calculate_connectivity(self, labels_volume, num_pores):
        """Calculate percentage of largest connected pore volume."""
        if num_pores == 0:
            return 0.0
        
        # Find volume of each pore
        pore_volumes = []
        for i in range(1, num_pores + 1):
            volume = np.sum(labels_volume == i)
            pore_volumes.append(volume)
        
        if len(pore_volumes) == 0:
            return 0.0
        
        largest_volume = max(pore_volumes)
        total_volume = sum(pore_volumes)
        
        return (largest_volume / total_volume) * 100.0 if total_volume > 0 else 0.0

    def _create_empty_result(self, data: VolumeData) -> VolumeData:
        return VolumeData(metadata={"Type": "Empty", "PoreCount": 0})