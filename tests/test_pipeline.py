import numpy as np
import pytest

from core import (
    VolumeData,
    VolumeProcessDTO,
    build_volume_pipeline,
    resolve_pipeline_stages,
    run_volume_pipeline,
)


def _make_sample_volume() -> VolumeData:
    data = np.ones((12, 12, 12), dtype=np.float32) * 1000
    data[4:8, 4:8, 4:8] = -1000
    return VolumeData(raw_data=data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), metadata={})


def test_resolve_pipeline_stages():
    assert resolve_pipeline_stages(target_stage="segment", include_export=False) == (
        "load",
        "threshold",
        "segment",
    )
    assert resolve_pipeline_stages(target_stage="pnm", include_export=False) == (
        "load",
        "threshold",
        "segment",
        "pnm",
    )


def test_resolve_pipeline_stages_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_pipeline_stages(target_stage="unknown")  # type: ignore[arg-type]


def test_build_volume_pipeline_for_segment():
    dto = VolumeProcessDTO(threshold=-300.0)
    dag = build_volume_pipeline(
        dto=dto,
        input_data=_make_sample_volume(),
        target_stage="segment",
        include_export=False,
    )
    assert set(dag._nodes.keys()) == {"load", "threshold", "segment"}


def test_run_pipeline_threshold_uses_preloaded_data():
    dto = VolumeProcessDTO(threshold=-300.0)
    volume = _make_sample_volume()
    results = run_volume_pipeline(
        dto=dto,
        input_data=volume,
        target_stage="threshold",
        include_export=False,
    )
    assert results["load"] is volume
    assert results["threshold"] == pytest.approx(-300.0)


def test_run_pipeline_segment_stage():
    dto = VolumeProcessDTO(threshold=0.0)
    results = run_volume_pipeline(
        dto=dto,
        input_data=_make_sample_volume(),
        target_stage="segment",
        include_export=False,
    )
    segmented = results.get("segment")
    assert segmented is not None
    assert segmented.raw_data is not None
    assert "pnm" not in results
