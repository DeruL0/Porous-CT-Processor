import numpy as np

from core import VolumeData
from data.manager import ScientificDataManager
from gui.handlers.timeseries_handler import TimeSeriesTransformWorker


def _volume(values) -> VolumeData:
    raw = np.array(values, dtype=np.float32)
    return VolumeData(raw_data=raw, metadata={"Type": "CT"})


def test_transform_worker_clip_applies_to_all_frames():
    volumes = [
        _volume([[[ -500, 0], [200, 800]], [[-300, 50], [100, 600]]]),
        _volume([[[ -900, -100], [20, 300]], [[-50, 10], [100, 250]]]),
    ]
    finished = []
    worker = TimeSeriesTransformWorker(
        volumes=volumes,
        mode="clip",
        current_index=0,
        min_val=-200.0,
        max_val=200.0,
    )
    worker.finished.connect(lambda payload: finished.append(payload))
    worker.run()

    assert len(finished) == 1
    assert finished[0]["mode"] == "clip"
    for vol in volumes:
        assert float(vol.raw_data.min()) >= -200.0
        assert float(vol.raw_data.max()) <= 200.0


def test_transform_worker_invert_returns_current_offset():
    volumes = [
        _volume([[[ -3, 1], [2, 5]], [[0, -1], [4, 3]]]),
        _volume([[[10, 20], [30, 40]], [[15, 25], [35, 45]]]),
    ]
    originals = [v.raw_data.copy() for v in volumes]
    finished = []
    worker = TimeSeriesTransformWorker(
        volumes=volumes,
        mode="invert",
        current_index=1,
    )
    worker.finished.connect(lambda payload: finished.append(payload))
    worker.run()

    assert len(finished) == 1
    payload = finished[0]
    assert payload["mode"] == "invert"

    for idx, vol in enumerate(volumes):
        expected_offset = float(originals[idx].min() + originals[idx].max())
        expected = expected_offset - originals[idx]
        np.testing.assert_allclose(vol.raw_data, expected)

    expected_current_offset = float(originals[1].min() + originals[1].max())
    assert payload["invert_offset"] == expected_current_offset


def test_transform_worker_cancel_stops_after_first_frame(monkeypatch):
    volumes = [
        _volume([[[ -500, 0], [200, 800]], [[-300, 50], [100, 600]]]),
        _volume([[[ -900, -100], [20, 300]], [[-50, 10], [100, 250]]]),
    ]
    original_second = volumes[1].raw_data.copy()
    worker = TimeSeriesTransformWorker(
        volumes=volumes,
        mode="clip",
        current_index=0,
        min_val=-200.0,
        max_val=200.0,
    )

    calls = {"count": 0}
    original_clip = ScientificDataManager.clip_volume_inplace

    def _wrapped_clip(volume, min_val, max_val):
        calls["count"] += 1
        original_clip(volume, min_val=min_val, max_val=max_val)
        if calls["count"] == 1:
            worker.cancel()

    monkeypatch.setattr(
        ScientificDataManager,
        "clip_volume_inplace",
        staticmethod(_wrapped_clip),
    )

    cancelled = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run()

    assert cancelled
    assert calls["count"] == 1
    assert float(volumes[0].raw_data.min()) >= -200.0
    np.testing.assert_allclose(volumes[1].raw_data, original_second)
