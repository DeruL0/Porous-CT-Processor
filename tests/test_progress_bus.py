from core.progress import (
    CancelFlagObserver,
    ProgressBus,
    ProgressEvent,
    StageProgressMapper,
)


def test_progress_bus_stage_and_dag_callbacks():
    events = []
    bus = ProgressBus().subscribe(lambda e: events.append(e))

    bus.stage_callback("segment")(55, "segmenting")
    bus.dag_callback()(20, "Running: segment")

    assert len(events) == 2
    assert isinstance(events[0], ProgressEvent)
    assert events[0].channel == "stage"
    assert events[0].stage == "segment"
    assert events[0].percent == 55
    assert events[1].channel == "dag"
    assert events[1].stage is None
    assert events[1].percent == 20


def test_stage_progress_mapper():
    mapper = StageProgressMapper(["load", "threshold", "segment", "pnm"])
    assert mapper.map("load", 50) == 12
    assert mapper.map("threshold", 100) == 50
    assert mapper.map("pnm", 100) == 100


def test_cancel_flag_observer_raises():
    bus = ProgressBus()
    bus.subscribe(CancelFlagObserver(lambda: True))
    cb = bus.stage_callback("load")

    raised = False
    try:
        cb(10, "loading")
    except InterruptedError:
        raised = True

    assert raised
