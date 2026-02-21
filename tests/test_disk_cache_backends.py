import tempfile

import numpy as np

from data.disk_cache import (
    MemmapStorageBackend,
    PickleStorageBackend,
    SegmentationCache,
    clear_all_caches,
    get_segmentation_cache,
    get_timeseries_pnm_cache,
)


def test_memmap_backend_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        backend = MemmapStorageBackend(cache_dir=tmp, prefix="test_mem")
        array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

        entry = backend.write("vol:key", array)
        loaded = backend.read(entry)

        assert loaded is not None
        assert loaded.shape == array.shape
        assert np.array_equal(np.asarray(loaded), array)
        del loaded

        backend.delete(entry)
        assert backend.read(entry) is None


def test_pickle_backend_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        backend = PickleStorageBackend(cache_dir=tmp, prefix="test_obj")
        payload = {"a": 1, "b": [1, 2, 3], "c": {"k": "v"}}

        entry = backend.write("session:key", payload)
        loaded = backend.read(entry)

        assert loaded == payload

        backend.delete(entry)
        assert backend.read(entry) is None


def test_segmentation_cache_disk_path_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        cache = SegmentationCache(cache_dir=tmp, memmap_threshold_mb=0)
        mask = (np.random.rand(8, 8, 8) > 0.5).astype(np.uint8)
        cache.store_pores_mask("vol1", mask, metadata={"porosity": 12.34}, use_disk=True)

        loaded = cache.get_pores_mask("vol1")
        assert loaded is not None
        assert np.array_equal(np.asarray(loaded), mask)
        del loaded
        assert cache.get_metadata("vol1") == {"porosity": 12.34}

        cache.clear("vol1")
        assert cache.get_pores_mask("vol1") is None


def test_clear_all_caches_clears_registered_singletons():
    seg = get_segmentation_cache()
    ts = get_timeseries_pnm_cache()

    seg.store_pores_mask("volX", np.ones((4, 4, 4), dtype=np.uint8), metadata={"m": 1}, use_disk=False)
    ts_entry = ts._backend.write("dummy", {"value": 1})
    ts._cache["dummy"] = {"backend_entry": ts_entry, "num_timepoints": 1, "num_pores": 1}
    assert ts.has_cache("dummy")

    clear_all_caches()

    assert seg.get_pores_mask("volX") is None
    assert "dummy" not in ts._cache
