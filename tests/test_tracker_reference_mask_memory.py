import numpy as np

import processors.pnm_tracker as pnm_tracker
from core.time_series import PNMSnapshot


def _make_snapshot(regions: np.ndarray, time_index: int = 0) -> PNMSnapshot:
    labels = np.unique(regions)
    labels = labels[labels > 0]
    centers = []
    radii = []
    volumes = []
    for label in labels:
        coords = np.argwhere(regions == label)
        centers.append(coords.mean(axis=0).astype(np.float64))
        volumes.append(float(len(coords)))
        r_vox = (3.0 * len(coords) / (4.0 * np.pi)) ** (1.0 / 3.0)
        radii.append(float(r_vox))

    return PNMSnapshot(
        time_index=time_index,
        pore_centers=np.asarray(centers, dtype=np.float64),
        pore_radii=np.asarray(radii, dtype=np.float64),
        pore_ids=np.asarray(labels, dtype=np.int32),
        pore_volumes=np.asarray(volumes, dtype=np.float64),
        connections=[],
        segmented_regions=regions,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        metadata={},
    )


def test_cache_reference_masks_uses_find_objects_not_argwhere(monkeypatch):
    regions = np.zeros((32, 32, 32), dtype=np.int32)
    label = 1
    for z in range(2, 30, 6):
        for y in range(2, 30, 6):
            for x in range(2, 30, 6):
                regions[z:z + 2, y:y + 2, x:x + 2] = label
                label += 1

    snapshot = _make_snapshot(regions, time_index=0)
    tracker = pnm_tracker.PNMTracker(
        match_mode="temporal_global",
        assign_solver="scipy",
        use_gpu=False,
        use_batch=False,
    )

    find_calls = {"count": 0}
    original_find_objects = pnm_tracker.ndimage.find_objects

    def _wrapped_find_objects(*args, **kwargs):
        find_calls["count"] += 1
        return original_find_objects(*args, **kwargs)

    def _forbid_argwhere(*_args, **_kwargs):
        raise AssertionError("np.argwhere should not be used in _cache_reference_masks")

    monkeypatch.setattr(pnm_tracker.ndimage, "find_objects", _wrapped_find_objects)
    monkeypatch.setattr(pnm_tracker.np, "argwhere", _forbid_argwhere)

    tracker._cache_reference_masks(snapshot)

    assert find_calls["count"] == 1
    assert len(tracker._reference_masks) == snapshot.num_pores

