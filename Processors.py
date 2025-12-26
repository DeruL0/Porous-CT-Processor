import numpy as np
import scipy.ndimage as ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from Core import BaseProcessor, VolumeData


# ==========================================
# Image Processor
# ==========================================

class PoreExtractionProcessor(BaseProcessor):
    """
    Extract void space (pores) from the solid matrix using morphological operations.
    """

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        print(f"[Processor] Starting pore detection (Threshold < {threshold} Intensity)...")

        # In Porous CT, pores are usually low intensity (air) and solid is high intensity.
        # We look for the "background" relative to the solid material.
        # Note: Logic assumes user passes a threshold where values BELOW or ABOVE define the pore.
        # Standard CT: Air ~ -1000, Bone/Rock ~ 1000.
        # Here we assume the user sets a threshold to separate solid from pore.

        # NOTE: Depending on input, we might need to invert logic.
        # Here we assume data.raw_data > threshold is SOLID.
        # Therefore, data.raw_data <= threshold is PORE.

        # To maintain compatibility with the original logic (which extracted "foreground"),
        # we will treat the selected region as the target "Pore" region.
        # Let's assume the user selects the range they want to visualize.

        target_mask = data.raw_data > threshold

        print("[Processor] Performing morphological operations (may take a few seconds)...")
        # Fill holes inside the selected region to get full volume
        filled_mask = ndimage.binary_fill_holes(target_mask)

        # XOR to find internal voids if we segmented the solid
        # OR if we segmented the pores directly, this logic might need adjustment.
        # For this tool, we assume: "Extract Pores" means identifying the void space.

        pores_mask = filled_mask ^ target_mask

        pore_voxels = np.sum(pores_mask)
        print(f"[Processor] Detection complete. Found {pore_voxels} pore voxels.")

        processed_volume = np.zeros_like(data.raw_data)
        processed_volume[pores_mask] = 1000  # Assign high value for visualization of the pore itself

        return VolumeData(
            raw_data=processed_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Void Space",
                "Source": data.metadata.get("Description", "Unknown"),
                "PoreVoxels": int(pore_voxels)
            }
        )


class PoreToSphereProcessor(BaseProcessor):
    """
    Detects pores and constructs a Pore Network Model (PNM) using Watershed Segmentation.
    Used for characterizing porous media (rocks, foams, bones, filters).

    Algorithm:
    1. Identify pore space.
    2. Compute Euclidean Distance Map (EDM) on the pore space.
    3. Find local maxima in the EDM to identify pore bodies (Seeds).
    4. Apply Watershed segmentation to split large connected pores into sub-pores.
    5. Determine connectivity (throats) by checking adjacency between segmented regions.
    6. Visualize as Ball-and-Stick model.
    """

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        print(f"[Processor] Starting Pore Network Extraction (Threshold > {threshold})...")

        # 1. Detect Pores (Binary Mask)
        foreground_mask = data.raw_data > threshold
        filled_mask = ndimage.binary_fill_holes(foreground_mask)
        pores_mask = filled_mask ^ foreground_mask

        if np.sum(pores_mask) == 0:
            return VolumeData(
                raw_data=np.zeros_like(data.raw_data),
                spacing=data.spacing,
                origin=data.origin,
                metadata={"Type": "Processed - Network", "PoreCount": 0}
            )

        # 2. Distance Transform
        # Calculate the distance from every pore voxel to the nearest solid/matrix voxel
        # The 'peaks' of this map represent the centers of wide open pore bodies
        print("[Processor] Computing Euclidean Distance Map...")
        distance_map = ndimage.distance_transform_edt(pores_mask)

        # 3. Find Pore Centers (Seeds)
        # We look for local peaks in the distance map.
        print("[Processor] Finding local maxima (seeds)...")
        min_peak_distance = 6
        local_maxi = peak_local_max(distance_map, labels=pores_mask, min_distance=min_peak_distance)

        # Create a marker array for watershed
        markers = np.zeros_like(distance_map, dtype=int)
        # peak_local_max returns coordinates (N, 3), we mark them on the grid
        markers[tuple(local_maxi.T)] = np.arange(len(local_maxi)) + 1

        print(f"[Processor] Found {len(local_maxi)} potential pore bodies.")

        # 4. Watershed Segmentation
        # "Floods" the topography (inverted distance map) starting from the markers.
        # This splits the continuous pore space into distinct regions.
        # The boundaries where basins meet are the 'throats'.
        print("[Processor] Executing Watershed segmentation to split large pores...")
        segmented_pores = watershed(-distance_map, markers, mask=pores_mask)

        num_pores = np.max(segmented_pores)
        print(f"[Processor] Segmentation complete. Identified {num_pores} unique pore regions.")

        # 5. Extract Spheres (Nodes)
        print("[Processor] Calculating pore properties...")
        sphere_volume = np.zeros_like(data.raw_data)

        slices = ndimage.find_objects(segmented_pores)
        spheres = {}  # Dict to store id -> {center, radius}

        for i in range(num_pores):
            label_idx = i + 1
            slice_obj = slices[i]
            if slice_obj is None: continue

            # Local extraction of the specific pore region
            local_mask = (segmented_pores[slice_obj] == label_idx)
            voxel_count = np.sum(local_mask)

            # Calculate equivalent radius of a sphere with the same volume
            radius = (3 * voxel_count / (4 * np.pi)) ** (1 / 3)

            # Calculate Centroid
            local_centroid = ndimage.center_of_mass(local_mask)
            global_z = slice_obj[0].start + local_centroid[0]
            global_y = slice_obj[1].start + local_centroid[1]
            global_x = slice_obj[2].start + local_centroid[2]

            center = np.array([global_z, global_y, global_x])

            # Store info
            spheres[label_idx] = {'center': center, 'radius': radius}

            # Draw Sphere (Node) into visualization volume
            self._draw_sphere(sphere_volume, center, radius)

        # 6. Detect Connectivity (Adjacency)
        # If two different labeled regions touch each other, they are connected via a throat.
        print("[Processor] Analyzing region adjacency to build network connections...")
        connections = self._find_adjacency(segmented_pores)
        print(f"[Processor] Found {len(connections)} connections (throats).")

        # 7. Draw Cylinders (Edges)
        print("[Processor] Drawing network connections...")
        for (id_a, id_b) in connections:
            if id_a in spheres and id_b in spheres:
                p1 = spheres[id_a]['center']
                p2 = spheres[id_b]['center']
                r1 = spheres[id_a]['radius']
                r2 = spheres[id_b]['radius']

                # Cylinder radius - heuristic:
                # It should represent the throat. We estimate it as a fraction of the smaller pore.
                cyl_radius = min(r1, r2) * 0.25
                cyl_radius = max(1.0, cyl_radius)  # Ensure visibility (at least 1 voxel)

                self._draw_cylinder(sphere_volume, p1, p2, cyl_radius)

        return VolumeData(
            raw_data=sphere_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Network (PNM)",
                "Source": data.metadata.get("Description", "Unknown"),
                "PoreCount": num_pores,
                "ConnectionCount": len(connections)
            }
        )

    def _find_adjacency(self, labels_volume):
        """
        Scans the volume to find voxels with different labels that are touching.
        Returns a set of tuples (label_a, label_b).
        """
        adjacency = set()

        # Directions to check: (axis, slice_current, slice_next)
        shifts = [
            (0, np.s_[:-1, :, :], np.s_[1:, :, :]),  # Z axis neighbors
            (1, np.s_[:, :-1, :], np.s_[:, 1:, :]),  # Y axis neighbors
            (2, np.s_[:, :, :-1], np.s_[:, :, 1:])  # X axis neighbors
        ]

        for axis, s_curr, s_next in shifts:
            val_curr = labels_volume[s_curr]
            val_next = labels_volume[s_next]

            mask = (val_curr > 0) & (val_next > 0) & (val_curr != val_next)

            if np.any(mask):
                pairs = np.stack((val_curr[mask], val_next[mask]), axis=-1)
                unique_pairs = np.unique(pairs, axis=0)

                for p in unique_pairs:
                    p_sorted = tuple(sorted(p))
                    adjacency.add(p_sorted)

        return adjacency

    def _draw_sphere(self, volume, center, radius):
        """Draws a solid sphere."""
        cz, cy, cx = center

        z_min = int(max(0, cz - radius - 1))
        z_max = int(min(volume.shape[0], cz + radius + 2))
        y_min = int(max(0, cy - radius - 1))
        y_max = int(min(volume.shape[1], cy + radius + 2))
        x_min = int(max(0, cx - radius - 1))
        x_max = int(min(volume.shape[2], cx + radius + 2))

        z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist_sq = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2

        mask = dist_sq <= radius ** 2
        volume[z_min:z_max, y_min:y_max, x_min:x_max][mask] = 1000

    def _draw_cylinder(self, volume, p1, p2, radius):
        """Draws a cylinder between point p1 and p2."""
        p_min = np.minimum(p1, p2) - radius - 1
        p_max = np.maximum(p1, p2) + radius + 2

        z_min = int(max(0, p_min[0]))
        z_max = int(min(volume.shape[0], p_max[0]))
        y_min = int(max(0, p_min[1]))
        y_max = int(min(volume.shape[1], p_max[1]))
        x_min = int(max(0, p_min[2]))
        x_max = int(min(volume.shape[2], p_max[2]))

        if z_min >= z_max or y_min >= y_max or x_min >= x_max:
            return

        vec = p2 - p1
        vec_len_sq = np.dot(vec, vec)

        if vec_len_sq == 0: return

        z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

        diff_z = z - p1[0]
        diff_y = y - p1[1]
        diff_x = x - p1[2]

        t = (diff_z * vec[0] + diff_y * vec[1] + diff_x * vec[2]) / vec_len_sq
        t = np.clip(t, 0.0, 1.0)

        closest_z = p1[0] + t * vec[0]
        closest_y = p1[1] + t * vec[1]
        closest_x = p1[2] + t * vec[2]

        dist_sq = (z - closest_z) ** 2 + (y - closest_y) ** 2 + (x - closest_x) ** 2

        mask = dist_sq <= radius ** 2
        volume[z_min:z_max, y_min:y_max, x_min:x_max][mask] = 1000