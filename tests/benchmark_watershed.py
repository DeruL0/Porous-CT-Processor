
import numpy as np
import time
from processors.utils import watershed_gpu
from skimage.segmentation import watershed as cpu_watershed
from scipy.ndimage import distance_transform_edt

def test_watershed():
    print("Creating synthetic data...")
    shape = (128, 128, 128)
    # Create two blobs
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (40, 64, 64)
    center2 = (88, 64, 64)
    mask1 = ((z-center1[0])**2 + (y-center1[1])**2 + (x-center1[2])**2) < 30**2
    mask2 = ((z-center2[0])**2 + (y-center2[1])**2 + (x-center2[2])**2) < 30**2
    image = np.zeros(shape, dtype=np.float32)
    image[mask1] = 1.0
    image[mask2] = 1.0
    
    # Distance transform (inverse for watershed)
    dist = distance_transform_edt(image).astype(np.float32)
    image_input = -dist
    
    # Markers
    markers = np.zeros(shape, dtype=np.int32)
    markers[center1] = 1
    markers[center2] = 2
    
    print("Running CPU watershed...")
    t0 = time.time()
    res_cpu = cpu_watershed(image_input, markers, mask=image>0)
    print(f"CPU Time: {time.time()-t0:.4f}s")
    
    print("Running GPU watershed...")
    try:
        t0 = time.time()
        res_gpu = watershed_gpu(image_input, markers, mask=image>0)
        print(f"GPU Time: {time.time()-t0:.4f}s")
        
        # Verify
        match = np.sum(res_cpu == res_gpu) / np.prod(shape)
        print(f"Match Similarity: {match*100:.2f}%")
        
        if match > 0.99:
            print("SUCCESS: GPU result matches CPU.")
        else:
            print("WARNING: Low similarity.")
            
    except Exception as e:
        print(f"GPU Failed: {e}")

if __name__ == "__main__":
    test_watershed()
