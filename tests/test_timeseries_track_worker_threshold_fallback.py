from types import SimpleNamespace

import numpy as np

from core import VolumeData
import gui.handlers.timeseries_handler as ts_handler_module
from gui.handlers.timeseries_handler import TimeSeriesTrackWorker


def _make_volume() -> VolumeData:
    return VolumeData(raw_data=np.ones((4, 4, 4), dtype=np.float32), metadata={"Type": "CT"})


def test_track_worker_keeps_user_threshold_when_t0_segmentation_empty(monkeypatch):
    call_log = {"extract_thresholds": [], "mesh_thresholds": []}

    class _DummySphereProcessor:
        def extract_snapshot(
            self,
            _volume,
            *,
            threshold: int,
            time_index: int,
            compute_connectivity: bool,
            callback=None,
        ):
            call_log["extract_thresholds"].append((time_index, int(threshold), bool(compute_connectivity)))
            if callback is not None:
                callback(50, "dummy")
            if time_index == 0 and int(threshold) == 30000:
                return SimpleNamespace(
                    segmented_regions=None,
                    num_pores=0,
                    connections=[],
                    pore_ids=np.array([], dtype=np.int32),
                    metadata={},
                )
            return SimpleNamespace(
                segmented_regions=np.ones((2, 2, 2), dtype=np.int32),
                num_pores=1,
                connections=[],
                pore_ids=np.array([1], dtype=np.int32),
                metadata={},
            )

        def process(self, _volume, *, threshold: int, callback=None):
            call_log["mesh_thresholds"].append(int(threshold))
            if callback is not None:
                callback(100, "dummy mesh")
            return "dummy_mesh"

    class _DummyTracker:
        def __init__(self):
            self._results = SimpleNamespace()

        def set_reference(self, snapshot):
            self._reference = snapshot

        def track_snapshot(self, snapshot):
            self._last = snapshot

        def get_results(self):
            return self._results

        def evaluate_against_sim_annotations(self, _volumes):
            return {}

    monkeypatch.setattr(ts_handler_module, "PoreToSphereProcessor", _DummySphereProcessor)
    monkeypatch.setattr(ts_handler_module, "PNMTracker", _DummyTracker)
    worker = TimeSeriesTrackWorker([_make_volume(), _make_volume()], threshold=30000)
    finished_payloads = []
    worker.finished.connect(lambda payload: finished_payloads.append(payload))
    worker.run()

    assert len(finished_payloads) == 1
    payload = finished_payloads[0]
    assert payload["resolved_threshold"] == 30000
    assert payload["threshold_autofixed"] is False
    assert call_log["extract_thresholds"] == [
        (0, 30000, True),
        (1, 30000, False),
    ]
    assert call_log["mesh_thresholds"] == [30000]
