import unittest
import numpy as np
from core import VolumeData
from processors import PoreExtractionProcessor, PoreToSphereProcessor

class TestProcessors(unittest.TestCase):
    def setUp(self):
        # Create a 50x50x50 volume with a central "pore" (cube of 0s in a field of 1s)
        self.size = 50
        self.data = np.ones((self.size, self.size, self.size), dtype=np.float32) * 1000 # Solid
        
        # Create a 10x10x10 void in the center
        center = self.size // 2
        start, end = center - 5, center + 5
        self.data[start:end, start:end, start:end] = -1000 # Air
        
        self.vol_data = VolumeData(raw_data=self.data, spacing=(1.0, 1.0, 1.0))

    def test_pore_extraction(self):
        processor = PoreExtractionProcessor()
        # Threshold: > 0 is Solid, < 0 is Air. 
        # Processor finds "Pores" (Air). Default threshold check might depend on logic.
        # Logic says: solid_mask = data > threshold. 
        # If threshold is -300: 
        #   1000 > -300 -> True (Solid)
        #   -1000 > -300 -> False (Air)
        # So pores are where False.
        
        result = processor.process(self.vol_data, threshold=0)
        
        # Metadata should contain porosity
        # 10x10x10 = 1000 voxels. Total = 125000. 
        # Porosity = 1000/125000 = 0.8%
        
        self.assertIn("Porosity", result.metadata)
        self.assertIn("PoreCount", result.metadata)
        
        # Check actual values (allow some float tolerance)
        p_count = result.metadata["PoreCount"]
        # It should find 1 connected component
        self.assertEqual(int(p_count), 1)

    def test_pnm_generation_caching(self):
        processor = PoreToSphereProcessor()
        
        # First Run
        result1 = processor.process(self.vol_data, threshold=0)
        self.assertTrue(result1.has_mesh)
        
        # Second Run (Should hit cache)
        # We can mock print or define a callback to verify, but here we just ensure it runs and returns same result
        result2 = processor.process(self.vol_data, threshold=0)
        
        self.assertEqual(result1.metadata["PoreCount"], result2.metadata["PoreCount"])
        self.assertEqual(result1.mesh.n_points, result2.mesh.n_points)

if __name__ == '__main__':
    unittest.main()
