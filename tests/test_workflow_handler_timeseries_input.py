import numpy as np

from core import VolumeData
from gui.handlers.workflow_handler import WorkflowHandler


class _DummyPanel:
    def setEnabled(self, _enabled: bool) -> None:
        return


class _DummyVisualizer:
    pass


class _DummyDataManager:
    def __init__(self, active_data):
        self.active_data = active_data


class _DummyTimeseriesHandler:
    def __init__(self, has_volumes: bool, current_volume):
        self.has_volumes = has_volumes
        self._current_volume = current_volume
        self.is_busy = False

    def get_current_raw_volume(self):
        return self._current_volume


class _DummyController:
    def __init__(self, active_data, ts_handler):
        self.visualizer = _DummyVisualizer()
        self.panel = _DummyPanel()
        self.data_manager = _DummyDataManager(active_data)
        self.timeseries_handler = ts_handler


def _make_volume(fill: float) -> VolumeData:
    raw = np.full((4, 4, 4), fill, dtype=np.float32)
    return VolumeData(raw_data=raw, metadata={"Type": "CT"})


def test_resolve_input_data_prefers_current_4d_timepoint():
    active = _make_volume(1.0)
    current_4d = _make_volume(2.0)
    controller = _DummyController(
        active_data=active,
        ts_handler=_DummyTimeseriesHandler(has_volumes=True, current_volume=current_4d),
    )
    handler = WorkflowHandler(controller)
    assert handler._resolve_input_data() is current_4d


def test_resolve_input_data_falls_back_to_active_data():
    active = _make_volume(1.0)
    controller = _DummyController(
        active_data=active,
        ts_handler=_DummyTimeseriesHandler(has_volumes=True, current_volume=None),
    )
    handler = WorkflowHandler(controller)
    assert handler._resolve_input_data() is active
