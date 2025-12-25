import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
from Core import BaseProcessor, VolumeData


# ==========================================
# Image Processor
# ==========================================

class PoreExtractionProcessor(BaseProcessor):
    """
    Extract inside pores
    """

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        print(f"[Processor] Starting pore detection (Foreground Threshold > {threshold} HU)...")

        foreground_mask = data.raw_data > threshold

        print("[Processor] Performing morphological operations (may take a few seconds)...")
        filled_mask = ndimage.binary_fill_holes(foreground_mask)

        # XOR operation to find the difference (the pores)
        pores_mask = filled_mask ^ foreground_mask

        pore_voxels = np.sum(pores_mask)
        print(f"[Processor] Detection complete. Found {pore_voxels} pore voxels.")

        processed_volume = np.zeros_like(data.raw_data)
        processed_volume[pores_mask] = 1000

        return VolumeData(
            raw_data=processed_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Pores",
                "Source": data.metadata.get("Description", "Unknown"),
                "PoreVoxels": int(pore_voxels)
            }
        )


class PoreToSphereProcessor(BaseProcessor):
    """
    Detects pores and converts each individual pore into an equivalent volume sphere.
    Also connects nearby spheres with cylinders to visualize the pore network (Ball-and-Stick model).
    """

    def process(self, data: VolumeData, threshold: int = -300) -> VolumeData:
        print(f"[Processor] Detecting pores for sphere & network generation (Threshold > {threshold} HU)...")

        # 1. Detect Pores
        foreground_mask = data.raw_data > threshold
        filled_mask = ndimage.binary_fill_holes(foreground_mask)
        pores_mask = filled_mask ^ foreground_mask

        # 2. Label connected components
        print("[Processor] Labeling connected components...")
        labeled_array, num_features = ndimage.label(pores_mask, structure=np.ones((3, 3, 3)))
        print(f"[Processor] Found {num_features} individual pores.")

        # 3. Create output volume
        sphere_volume = np.zeros_like(data.raw_data)

        if num_features == 0:
            return VolumeData(
                raw_data=sphere_volume,
                spacing=data.spacing,
                origin=data.origin,
                metadata={"Type": "Processed - Network", "PoreCount": 0}
            )

        # 4. Extract Spheres (Nodes)
        print("[Processor] Calculating equivalent spheres...")
        slices = ndimage.find_objects(labeled_array)
        spheres = []  # List to store {'center': (z,y,x), 'radius': r}

        for i in range(num_features):
            label_idx = i + 1
            slice_obj = slices[i]

            local_mask = (labeled_array[slice_obj] == label_idx)
            voxel_count = np.sum(local_mask)
            radius = (3 * voxel_count / (4 * np.pi)) ** (1 / 3)

            local_centroid = ndimage.center_of_mass(local_mask)
            global_z = slice_obj[0].start + local_centroid[0]
            global_y = slice_obj[1].start + local_centroid[1]
            global_x = slice_obj[2].start + local_centroid[2]

            center = np.array([global_z, global_y, global_x])

            # Draw sphere
            self._draw_sphere(sphere_volume, center, radius)

            # Store for networking
            spheres.append({'center': center, 'radius': radius})

        # 5. Generate Network (Sticks/Cylinders)
        print("[Processor] Generating pore network connections (Cylinders)...")
        #self._connect_pores_with_cylinders(sphere_volume, spheres)

        print("[Processor] Pore Network Model generation complete.")

        return VolumeData(
            raw_data=sphere_volume,
            spacing=data.spacing,
            origin=data.origin,
            metadata={
                "Type": "Processed - Network",
                "Source": data.metadata.get("Description", "Unknown"),
                "PoreCount": num_features
            }
        )

    def _draw_sphere(self, volume, center, radius):
        """Draws a solid sphere."""
        cz, cy, cx = center

        # Bounding box
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

    def _connect_pores_with_cylinders(self, volume, spheres, k_neighbors=3):
        """
        Connects nearby spheres with cylinders using K-Nearest Neighbors logic.
        """
        if len(spheres) < 2: return

        centers = [s['center'] for s in spheres]
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(centers)

        # Query k+1 neighbors (because the first one is the point itself)
        k = min(len(spheres), k_neighbors + 1)
        distances, indices = tree.query(centers, k=k)

        drawn_pairs = set()

        for i, neighbors in enumerate(indices):
            p1 = centers[i]
            r1 = spheres[i]['radius']

            for j, neighbor_idx in enumerate(neighbors):
                if i == neighbor_idx: continue  # Skip self
                if neighbor_idx >= len(spheres): continue

                # Check if pair already processed
                pair_key = tuple(sorted((i, neighbor_idx)))
                if pair_key in drawn_pairs: continue
                drawn_pairs.add(pair_key)

                p2 = centers[neighbor_idx]
                r2 = spheres[neighbor_idx]['radius']
                dist = distances[i][j]

                # Distance Threshold Heuristic:
                # Only connect if they are reasonably close relative to their sizes.
                # e.g., if distance < 5 * (sum of radii), assume connected
                if dist > (r1 + r2) * 5.0 + 10:
                    continue

                # Cylinder radius is proportional to the smaller pore
                cyl_radius = min(r1, r2) * 0.3
                if cyl_radius < 1.0: cyl_radius = 1.0

                self._draw_cylinder(volume, p1, p2, cyl_radius)

    def _draw_cylinder(self, volume, p1, p2, radius):
        """
        Draws a cylinder between point p1 and p2 with given radius.
        """
        # Calculate bounding box for the cylinder to limit iteration
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

        # Vector from p1 to p2
        vec = p2 - p1
        vec_len_sq = np.dot(vec, vec)

        if vec_len_sq == 0: return

        # Grid coordinate generation
        z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

        # Vector from p1 to current point
        # Note: We need to stack/broadcast carefully manually or use logic below
        # For efficiency in Python without large loops, we do component-wise calc:

        # This calculates the projection t of point X onto line segment P1-P2
        # t = dot(X - P1, P2 - P1) / |P2 - P1|^2

        diff_z = z - p1[0]
        diff_y = y - p1[1]
        diff_x = x - p1[2]

        t = (diff_z * vec[0] + diff_y * vec[1] + diff_x * vec[2]) / vec_len_sq

        # Clamp t to segment [0, 1]
        t = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        closest_z = p1[0] + t * vec[0]
        closest_y = p1[1] + t * vec[1]
        closest_x = p1[2] + t * vec[2]

        # Distance from point to closest point on segment
        dist_sq = (z - closest_z) ** 2 + (y - closest_y) ** 2 + (x - closest_x) ** 2

        mask = dist_sq <= radius ** 2
        volume[z_min:z_max, y_min:y_max, x_min:x_max][mask] = 1000