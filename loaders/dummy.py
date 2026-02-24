"""
Synthetic data generators for testing.
"""

import numpy as np
from typing import Optional, Callable

from core import BaseLoader, VolumeData


class DummyLoader(BaseLoader):
    """Synthetic porous media generator for testing"""

    def load(self, size: int = 128, callback: Optional[Callable[[int, str], None]] = None) -> VolumeData:
        print(f"[Loader] Generating synthetic porous structure with random spheres (size={size})...")
        rng = np.random.default_rng()
        if callback:
            callback(0, "Initializing solid matrix...")

        # 1) Start from a full solid block.
        volume = np.full((size, size, size), 1000.0, dtype=np.float32)

        # 2) Carve spherical voids inside the interior.
        border = max(1, min(5, int(size) // 6))
        interior = max(1, size - 2 * border)
        min_radius = max(1, interior // 24)
        max_radius = max(min_radius, interior // 10)
        target_void_fraction = 0.12
        target_void_voxels = int((interior ** 3) * target_void_fraction)
        max_spheres = max(8, min(256, interior * 2))

        if callback:
            callback(20, "Carving random spherical pores...")

        carved_voxels = 0
        inserted_spheres = 0
        for i in range(max_spheres):
            radius = int(rng.integers(min_radius, max_radius + 1))
            low = border + radius
            high = size - border - radius - 1
            if low > high:
                continue

            cz = int(rng.integers(low, high + 1))
            cy = int(rng.integers(low, high + 1))
            cx = int(rng.integers(low, high + 1))

            z0, z1 = cz - radius, cz + radius + 1
            y0, y1 = cy - radius, cy + radius + 1
            x0, x1 = cx - radius, cx + radius + 1
            local = volume[z0:z1, y0:y1, x0:x1]

            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            sphere_mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

            already_void = np.count_nonzero(local[sphere_mask] < 0)
            local[sphere_mask] = -1000.0
            carved_voxels += int(np.count_nonzero(sphere_mask) - already_void)
            inserted_spheres += 1

            if callback and (i % 8 == 0 or carved_voxels >= target_void_voxels):
                progress = 20 + int(55 * min(1.0, carved_voxels / max(1, target_void_voxels)))
                callback(progress, f"Carving spherical pores ({inserted_spheres}/{max_spheres})...")

            if carved_voxels >= target_void_voxels:
                break

        if inserted_spheres == 0:
            # Fallback for very small volumes: force one center sphere.
            radius = max(1, interior // 4)
            cz = cy = cx = size // 2
            z0, z1 = max(0, cz - radius), min(size, cz + radius + 1)
            y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            sphere_mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            volume[z0:z1, y0:y1, x0:x1][sphere_mask] = -1000.0
            inserted_spheres = 1

        # 3) Enforce solid shell boundary.
        print("[Loader] Enforcing solid boundary shell...")
        if callback:
            callback(85, "Adding boundary shell...")
        volume[:border, :, :] = 1000
        volume[-border:, :, :] = 1000
        volume[:, :border, :] = 1000
        volume[:, -border:, :] = 1000
        volume[:, :, :border] = 1000
        volume[:, :, -border:] = 1000

        if callback:
            callback(100, "Generation complete.")

        actual_void_voxels = int(np.count_nonzero(volume < 0))
        actual_void_ratio = float(actual_void_voxels / volume.size)
        return VolumeData(
            raw_data=volume,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            metadata={
                "Type": "Synthetic",
                "Description": "Solid block with internal random spherical pores",
                "GenerationMethod": "Random Sphere Insertion + Solid Shell",
                "InsertedSphereCount": int(inserted_spheres),
                "VoidVoxelRatio": actual_void_ratio,
            },
        )
