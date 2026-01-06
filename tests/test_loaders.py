"""
Unit tests for DICOM loaders.
"""

import unittest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.dicom import _natural_sort_key


class TestNaturalSortKey(unittest.TestCase):
    """Test natural sorting function for filenames."""
    
    def test_numeric_sorting(self):
        """Test that numeric parts are sorted numerically, not lexicographically."""
        files = ['img_1.dcm', 'img_10.dcm', 'img_2.dcm', 'img_20.dcm', 'img_3.dcm']
        sorted_files = sorted(files, key=_natural_sort_key)
        expected = ['img_1.dcm', 'img_2.dcm', 'img_3.dcm', 'img_10.dcm', 'img_20.dcm']
        self.assertEqual(sorted_files, expected)
    
    def test_case_insensitive(self):
        """Test that sorting is case-insensitive."""
        files = ['IMG_1.dcm', 'img_2.dcm', 'Img_3.dcm']
        sorted_files = sorted(files, key=_natural_sort_key)
        expected = ['IMG_1.dcm', 'img_2.dcm', 'Img_3.dcm']
        self.assertEqual(sorted_files, expected)
    
    def test_mixed_format(self):
        """Test various filename formats."""
        files = ['slice001.dcm', 'slice010.dcm', 'slice002.dcm']
        sorted_files = sorted(files, key=_natural_sort_key)
        expected = ['slice001.dcm', 'slice002.dcm', 'slice010.dcm']
        self.assertEqual(sorted_files, expected)
    
    def test_pure_numbers(self):
        """Test files named with just numbers."""
        files = ['1', '10', '2', '20', '100']
        sorted_files = sorted(files, key=_natural_sort_key)
        expected = ['1', '2', '10', '20', '100']
        self.assertEqual(sorted_files, expected)


class TestLoaderImports(unittest.TestCase):
    """Test that loader classes can be imported correctly."""
    
    def test_import_loaders(self):
        """Test that all loader classes can be imported."""
        from loaders.dicom import DicomSeriesLoader, FastDicomLoader
        from loaders.dicom import MemoryMappedDicomLoader, ChunkedDicomLoader
        
        self.assertIsNotNone(DicomSeriesLoader)
        self.assertIsNotNone(FastDicomLoader)
        self.assertIsNotNone(MemoryMappedDicomLoader)
        self.assertIsNotNone(ChunkedDicomLoader)
    
    def test_loader_initialization(self):
        """Test that loaders can be instantiated with default args."""
        from loaders.dicom import DicomSeriesLoader, FastDicomLoader
        
        loader1 = DicomSeriesLoader()
        self.assertEqual(loader1.use_header_sort, True)  # Default is now True
        self.assertEqual(loader1.max_workers, 4)
        
        loader2 = DicomSeriesLoader(use_header_sort=False, max_workers=8)
        self.assertEqual(loader2.use_header_sort, False)
        self.assertEqual(loader2.max_workers, 8)
        
        fast_loader = FastDicomLoader(step=4)
        self.assertEqual(fast_loader.step, 4)


if __name__ == '__main__':
    unittest.main()
