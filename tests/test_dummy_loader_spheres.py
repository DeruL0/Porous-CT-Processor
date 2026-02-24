import numpy as np

from loaders import DummyLoader


def test_dummy_loader_generates_binary_sphere_based_volume():
    loader = DummyLoader()
    vol = loader.load(size=48)

    assert vol.raw_data is not None
    arr = vol.raw_data
    assert arr.shape == (48, 48, 48)

    unique = np.unique(arr)
    assert set(unique.tolist()) <= {-1000.0, 1000.0}
    assert np.any(arr < 0), "Expected void voxels from sphere carving"
    assert np.any(arr > 0), "Expected solid matrix voxels"

    border = max(1, min(5, 48 // 6))
    assert np.all(arr[:border, :, :] == 1000.0)
    assert np.all(arr[-border:, :, :] == 1000.0)
    assert np.all(arr[:, :border, :] == 1000.0)
    assert np.all(arr[:, -border:, :] == 1000.0)
    assert np.all(arr[:, :, :border] == 1000.0)
    assert np.all(arr[:, :, -border:] == 1000.0)

    assert vol.metadata.get("GenerationMethod") == "Random Sphere Insertion + Solid Shell"
    assert int(vol.metadata.get("InsertedSphereCount", 0)) > 0
