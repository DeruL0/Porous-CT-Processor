import numpy as np
import scipy.ndimage as ndimage
import pyvista as pv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Tuple, Dict, Set
from Core import BaseProcessor, VolumeData


# ==========================================
# Image Processors
# ==========================================

class PoreExtractionProcessor(BaseProcessor):
    """
    Segments the void space (pores) from the solid matrix.
    Returns a Voxel-based VolumeData object representing "Air".
    """

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        print(f"[Processor] Starting pore detection (Threshold < {threshold})...")

        # 1. Binarization (Air vs Solid)
        solid_mask = data.raw_data > threshold

        print("[Processor] Filling holes...")
        filled_volume = ndimage.binary_fill_holes(solid_mask)
        pores_mask = filled_volume ^ solid_mask

        # Quantitative Analysis
        pore_voxels = np.sum(pores_mask)
        total_voxels = data.raw_data.size
        porosity_pct = (pore_voxels / total_voxels) * 100.0

        # Label connected components
        labeled_array, num_features = ndimage.label(pores_mask, structure=np.ones((3, 3, 3)))

        print(f"[Processor] Found {num_features} pores.")

        # Create Output Volume (Voxel Grid)
        processed_volume = np.zeros_like(data.raw_data)
        processed_volume[pores_mask] = 1000  # Highlight Pores

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

    Improvements:
    1. Direct Mesh Generation: Creates pyvista.PolyData directly (no slow rasterization).
    2. Data Classification: Adds 'IsPore' boolean field to distinguish Pores (1) from Throats (0).
    """

    MIN_PEAK_DISTANCE = 6

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        if data.raw_data is None:
            raise ValueError("Input data must contain raw voxel data.")

        print(f"[PNM] Starting Mesh Extraction (Threshold > {threshold})...")

        # --- Step 1: Segmentation (Same as Extraction) ---
        solid_mask = data.raw_data > threshold
        filled_mask = ndimage.binary_fill_holes(solid_mask)
        pores_mask = filled_mask ^ solid_mask

        if np.sum(pores_mask) == 0:
            return self._create_empty_result(data)

        # --- Step 2: Watershed Analysis ---
        print("[PNM] Computing Distance Map & Watershed...")
        distance_map = ndimage.distance_transform_edt(pores_mask)

        local_maxi = peak_local_max(
            distance_map,
            labels=pores_mask,
            min_distance=self.MIN_PEAK_DISTANCE
        )

        markers = np.zeros_like(distance_map, dtype=int)
        markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1

        segmented_regions = watershed(-distance_map, markers, mask=pores_mask)
        num_pores = np.max(segmented_regions)

        # --- Step 3: Mesh Generation (Optimization) ---
        print("[PNM] Generating optimized mesh structures...")

        # A. Nodes (Pores) -> Spheres
        pore_centers, pore_radii, pore_ids = self._extract_pore_data(segmented_regions, num_pores, data.spacing,
                                                                     data.origin)

        # Create PyVista PolyData for Pores
        if len(pore_centers) > 0:
            pore_cloud = pv.PolyData(pore_centers)
            pore_cloud["radius"] = pore_radii
            # IsPore = 1 (True)
            pore_cloud["IsPore"] = np.ones(len(pore_centers), dtype=int)
            pore_cloud["ID"] = pore_ids

            # Efficient Glyphing: Scale a unit sphere by the radius array
            sphere_glyph = pv.Sphere(theta_resolution=10, phi_resolution=10)
            pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
        else:
            pores_mesh = pv.PolyData()

        # B. Edges (Throats) -> Tubes
        connections = self._find_adjacency(segmented_regions)
        throats_mesh = self._create_throat_mesh(connections, pore_centers, pore_radii, pore_ids)

        # --- Step 4: Combine ---
        print("[PNM] Merging geometry...")
        # Combine Pores and Throats into one mesh
        combined_mesh = pores_mesh.merge(throats_mesh)

        return VolumeData(
            raw_data=None,  # No Voxel data in this result, purely Mesh
            mesh=combined_mesh,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - PNM Mesh",
                "PoreCount": int(num_pores),
                "ConnectionCount": len(connections),
                "MeshPoints": combined_mesh.n_points
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
        """Creates tubes connecting the pores."""
        if not connections:
            return pv.PolyData()

        # Map ID to Index in 'centers' array
        id_map = {uid: idx for idx, uid in enumerate(ids_list)}

        lines = []

        for id_a, id_b in connections:
            if id_a in id_map and id_b in id_map:
                idx_a = id_map[id_a]
                idx_b = id_map[id_b]

                p1 = centers[idx_a]
                p2 = centers[idx_b]

                # Estimate throat radius (smaller than pores)
                r_throat = min(radii[idx_a], radii[idx_b]) * 0.3

                # Create a single tube
                line = pv.Line(p1, p2)
                tube = line.tube(radius=r_throat)
                lines.append(tube)

        if not lines:
            return pv.PolyData()

        # Merge all tubes
        throats = lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]

        # IsPore = 0 (False) -> This is a Throat
        throats["IsPore"] = np.zeros(throats.n_points, dtype=int)

        return throats

    def _create_empty_result(self, data: VolumeData) -> VolumeData:
        return VolumeData(metadata={"Type": "Empty", "PoreCount": 0})