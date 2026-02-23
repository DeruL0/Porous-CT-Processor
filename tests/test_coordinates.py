import unittest
import numpy as np

from core import VolumeData
from core.coordinates import (
    raw_zyx_to_grid_xyz,
    world_xyz_to_index_zyx,
    world_xyz_to_voxel_zyx,
    bounds_xyz_to_slices_zyx,
    origin_xyz_for_subvolume_zyx,
)
from rendering.roi_extractor import extract_box


class TestCoordinateConversions(unittest.TestCase):
    def test_raw_zyx_to_grid_xyz(self):
        raw = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
        grid = raw_zyx_to_grid_xyz(raw)
        self.assertEqual(grid.shape, (4, 3, 2))
        self.assertEqual(grid[3, 2, 1], raw[1, 2, 3])

    def test_world_to_voxel_and_index_zyx(self):
        spacing = (2.0, 3.0, 4.0)
        origin = (10.0, 20.0, 30.0)
        world = (14.4, 23.2, 34.1)

        zf, yf, xf = world_xyz_to_voxel_zyx(world, spacing, origin)
        self.assertAlmostEqual(zf, (34.1 - 30.0) / 4.0)
        self.assertAlmostEqual(yf, (23.2 - 20.0) / 3.0)
        self.assertAlmostEqual(xf, (14.4 - 10.0) / 2.0)

        z, y, x = world_xyz_to_index_zyx(world, spacing, origin, rounding="round")
        self.assertEqual((z, y, x), (1, 1, 2))

    def test_bounds_and_subvolume_origin(self):
        raw_shape = (5, 6, 7)  # z, y, x
        spacing = (2.0, 3.0, 4.0)
        origin = (10.0, 20.0, 30.0)
        bounds = (14.0, 20.0, 23.0, 29.0, 34.0, 42.0)

        z0, z1, y0, y1, x0, x1 = bounds_xyz_to_slices_zyx(bounds, raw_shape, spacing, origin)
        self.assertEqual((z0, z1, y0, y1, x0, x1), (1, 3, 1, 3, 2, 5))

        new_origin = origin_xyz_for_subvolume_zyx(origin, spacing, z0, y0, x0)
        self.assertEqual(new_origin, (14.0, 23.0, 34.0))


class TestRoiExtractionCoordinateConsistency(unittest.TestCase):
    def test_extract_box_uses_xyz_to_zyx_mapping(self):
        raw = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)  # z, y, x
        data = VolumeData(
            raw_data=raw,
            spacing=(2.0, 3.0, 4.0),
            origin=(10.0, 20.0, 30.0),
            metadata={"Type": "Test"},
        )
        bounds = (14.0, 20.0, 23.0, 29.0, 34.0, 42.0)

        out = extract_box(data, bounds)
        self.assertIsNotNone(out)
        self.assertEqual(out.raw_data.shape, (2, 2, 3))  # z, y, x
        self.assertEqual(out.origin, (14.0, 23.0, 34.0))
        np.testing.assert_array_equal(out.raw_data, raw[1:3, 1:3, 2:5])


if __name__ == "__main__":
    unittest.main()
