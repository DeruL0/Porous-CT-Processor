"""
Debug script for DICOM loader testing.
Run this to diagnose loading issues.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from loaders.dicom import DicomSeriesLoader, FastDicomLoader

def debug_load(folder_path: str):
    """Test DICOM loading with debug output."""
    print("=" * 60)
    print(f"Testing folder: {folder_path}")
    print("=" * 60)
    
    # Test with standard loader
    print("\n[TEST 1] DicomSeriesLoader (default, use_header_sort=True)")
    try:
        loader = DicomSeriesLoader()
        data = loader.load(folder_path)
        print(f"  Shape: {data.raw_data.shape}")
        print(f"  Spacing: {data.spacing}")
        print(f"  Value range: {np.nanmin(data.raw_data):.1f} to {np.nanmax(data.raw_data):.1f}")
        print(f"  Metadata: {data.metadata}")
        
        # Check if data has variation (not just a solid block)
        std = np.nanstd(data.raw_data)
        print(f"  Std deviation: {std:.2f}")
        if std < 1.0:
            print("  ⚠️ WARNING: Very low variation - might be solid block!")
        else:
            print("  ✓ Data has good variation")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test with fast loader for comparison
    print("\n[TEST 2] FastDicomLoader (step=2)")
    try:
        loader = FastDicomLoader(step=2)
        data = loader.load(folder_path)
        print(f"  Shape: {data.raw_data.shape}")
        print(f"  Value range: {np.nanmin(data.raw_data):.1f} to {np.nanmax(data.raw_data):.1f}")
        
        std = np.nanstd(data.raw_data)
        print(f"  Std deviation: {std:.2f}")
        if std < 1.0:
            print("  ⚠️ WARNING: Very low variation - might be solid block!")
        else:
            print("  ✓ Data has good variation")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_loader.py <path_to_dicom_folder>")
        sys.exit(1)
    
    folder = sys.argv[1]
    debug_load(folder)
