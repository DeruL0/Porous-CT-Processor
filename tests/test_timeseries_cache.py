"""
Test time series PNM caching functionality.
"""

import numpy as np
from core import VolumeData
from core.time_series import PNMSnapshot, TimeSeriesPNM, PoreTrackingResult
from data import get_timeseries_pnm_cache, clear_timeseries_pnm_cache


def create_dummy_volume(shape=(50, 50, 50), folder_name="t0"):
    """Create a dummy volume for testing."""
    data = np.random.randint(-1000, 1000, shape, dtype=np.int16)
    volume = VolumeData()
    volume.raw_data = data
    volume.spacing = (1.0, 1.0, 1.0)
    volume.origin = (0.0, 0.0, 0.0)
    volume.metadata = {'folder_name': folder_name}
    return volume


def create_dummy_snapshot(time_index=0, num_pores=10):
    """Create a dummy PNM snapshot for testing."""
    pore_centers = np.random.rand(num_pores, 3) * 50
    pore_radii = np.random.rand(num_pores) * 2 + 1
    pore_ids = np.arange(1, num_pores + 1)
    pore_volumes = np.random.randint(100, 1000, num_pores)
    connections = [(i, i+1) for i in range(1, num_pores)]
    
    return PNMSnapshot(
        time_index=time_index,
        pore_centers=pore_centers,
        pore_radii=pore_radii,
        pore_ids=pore_ids,
        pore_volumes=pore_volumes,
        connections=connections,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        metadata={'threshold': -300}
    )


def create_dummy_timeseries_pnm(num_timepoints=3, num_pores=10):
    """Create a dummy TimeSeriesPNM for testing."""
    ts_pnm = TimeSeriesPNM()
    
    # Create reference snapshot
    reference = create_dummy_snapshot(0, num_pores)
    ts_pnm.reference_snapshot = reference
    ts_pnm.snapshots.append(reference)
    
    # Create tracking result
    ts_pnm.tracking = PoreTrackingResult(
        reference_ids=reference.pore_ids.tolist()
    )
    
    # Initialize tracking data
    for pore_id, volume in zip(reference.pore_ids, reference.pore_volumes):
        ts_pnm.tracking.volume_history[int(pore_id)] = [float(volume)]
    
    # Add more timepoints
    for t in range(1, num_timepoints):
        snapshot = create_dummy_snapshot(t, num_pores)
        ts_pnm.snapshots.append(snapshot)
        
        # Update tracking
        for pore_id, volume in zip(snapshot.pore_ids, snapshot.pore_volumes):
            ts_pnm.tracking.volume_history[int(pore_id)].append(float(volume))
    
    return ts_pnm


def test_cache_store_and_retrieve():
    """Test storing and retrieving time series PNM from cache."""
    print("\n=== Test: Cache Store and Retrieve ===")
    
    # Clear cache first
    clear_timeseries_pnm_cache()
    
    # Create test data
    volumes = [create_dummy_volume(folder_name=f"t{i}") for i in range(3)]
    threshold = -300
    
    ts_pnm = create_dummy_timeseries_pnm(num_timepoints=3)
    reference_mesh = create_dummy_volume(folder_name="mesh")
    reference_snapshot = ts_pnm.reference_snapshot
    
    # Get cache and generate key
    cache = get_timeseries_pnm_cache()
    cache_key = cache.generate_key(volumes, threshold)
    
    print(f"Cache key: {cache_key}")
    
    # Store data
    cache.store(cache_key, ts_pnm, reference_mesh, reference_snapshot)
    print("✓ Data stored to cache")
    
    # Verify cache exists
    assert cache.has_cache(cache_key), "Cache should exist after storing"
    print("✓ Cache exists")
    
    # Retrieve data
    cached_data = cache.get(cache_key)
    assert cached_data is not None, "Should retrieve cached data"
    
    cached_ts_pnm, cached_mesh, cached_snapshot = cached_data
    print("✓ Data retrieved from cache")
    
    # Verify data integrity
    assert cached_ts_pnm.num_timepoints == ts_pnm.num_timepoints
    assert cached_ts_pnm.num_reference_pores == ts_pnm.num_reference_pores
    print(f"✓ Data integrity verified: {cached_ts_pnm.num_timepoints} timepoints, "
          f"{cached_ts_pnm.num_reference_pores} pores")
    
    print("\n✅ Test passed!\n")


def test_cache_key_uniqueness():
    """Test that different configurations generate different keys."""
    print("\n=== Test: Cache Key Uniqueness ===")
    
    cache = get_timeseries_pnm_cache()
    
    # Same volumes, same threshold
    volumes1 = [create_dummy_volume(folder_name=f"t{i}") for i in range(3)]
    key1a = cache.generate_key(volumes1, -300)
    key1b = cache.generate_key(volumes1, -300)
    
    assert key1a == key1b, "Same volumes and threshold should generate same key"
    print(f"✓ Same config generates same key: {key1a[:8]}...")
    
    # Same volumes, different threshold
    key2 = cache.generate_key(volumes1, -400)
    assert key1a != key2, "Different threshold should generate different key"
    print(f"✓ Different threshold generates different key: {key2[:8]}...")
    
    # Different volumes, same threshold
    volumes2 = [create_dummy_volume(folder_name=f"t{i}") for i in range(3)]
    key3 = cache.generate_key(volumes2, -300)
    assert key1a != key3, "Different volumes should generate different key"
    print(f"✓ Different volumes generate different key: {key3[:8]}...")
    
    print("\n✅ Test passed!\n")


def test_cache_clear():
    """Test cache clearing functionality."""
    print("\n=== Test: Cache Clear ===")
    
    cache = get_timeseries_pnm_cache()
    clear_timeseries_pnm_cache()
    
    # Create and store test data
    volumes = [create_dummy_volume(folder_name=f"t{i}") for i in range(2)]
    cache_key = cache.generate_key(volumes, -300)
    
    ts_pnm = create_dummy_timeseries_pnm(num_timepoints=2)
    reference_mesh = create_dummy_volume()
    
    cache.store(cache_key, ts_pnm, reference_mesh, ts_pnm.reference_snapshot)
    print("✓ Data stored")
    
    assert cache.has_cache(cache_key), "Cache should exist"
    
    # Clear specific cache
    cache.clear(cache_key)
    assert not cache.has_cache(cache_key), "Cache should be cleared"
    print("✓ Specific cache cleared")
    
    # Store again and clear all
    cache.store(cache_key, ts_pnm, reference_mesh, ts_pnm.reference_snapshot)
    clear_timeseries_pnm_cache()
    assert not cache.has_cache(cache_key), "All cache should be cleared"
    print("✓ All cache cleared")
    
    print("\n✅ Test passed!\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Time Series PNM Cache Tests")
    print("="*60)
    
    try:
        test_cache_store_and_retrieve()
        test_cache_key_uniqueness()
        test_cache_clear()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
