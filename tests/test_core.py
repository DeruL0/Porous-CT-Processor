import unittest
import numpy as np
from core import VolumeData

class TestVolumeData(unittest.TestCase):
    def test_initialization(self):
        data = np.zeros((10, 10, 10))
        vol = VolumeData(raw_data=data, spacing=(1.0, 1.0, 1.0))
        self.assertIsNotNone(vol.raw_data)
        self.assertEqual(vol.dimensions, (10, 10, 10))
        self.assertFalse(vol.has_mesh)

    def test_metadata(self):
        vol = VolumeData(metadata={"Type": "Test"})
        self.assertEqual(vol.metadata["Type"], "Test")
        self.assertEqual(vol.dimensions, (0, 0, 0))

if __name__ == '__main__':
    unittest.main()
