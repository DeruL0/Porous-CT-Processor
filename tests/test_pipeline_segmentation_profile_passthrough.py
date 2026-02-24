import numpy as np

import processors
from core import VolumeData, VolumeProcessDTO, run_volume_pipeline


def _sample_volume() -> VolumeData:
    arr = np.ones((10, 10, 10), dtype=np.float32) * 1000.0
    arr[3:7, 3:7, 3:7] = -1000.0
    return VolumeData(raw_data=arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), metadata={})


def test_pipeline_passes_segmentation_profile_to_processors(monkeypatch):
    seen = {}

    class _DummyPoreProcessor:
        def process(self, data, callback=None, **kwargs):
            seen["segment_kwargs"] = dict(kwargs)
            out = np.zeros_like(data.raw_data, dtype=np.float32)
            out[2:4, 2:4, 2:4] = 1000.0
            return VolumeData(
                raw_data=out,
                spacing=data.spacing,
                origin=data.origin,
                metadata={"Type": "Processed - Void Volume"},
            )

    class _DummyPNMProcessor:
        def process(self, data, callback=None, **kwargs):
            seen["pnm_kwargs"] = dict(kwargs)
            return VolumeData(
                raw_data=None,
                spacing=data.spacing,
                origin=data.origin,
                metadata={"Type": "Processed - PNM Mesh"},
            )

    monkeypatch.setattr(processors, "PoreExtractionProcessor", lambda: _DummyPoreProcessor())
    monkeypatch.setattr(processors, "PoreToSphereProcessor", lambda: _DummyPNMProcessor())

    dto = VolumeProcessDTO(
        threshold=123.0,
        segmentation_profile="legacy",
        split_mode="aggressive",
    )
    run_volume_pipeline(
        dto=dto,
        input_data=_sample_volume(),
        target_stage="pnm",
        include_export=False,
    )

    assert seen["segment_kwargs"]["segmentation_profile"] == "legacy"
    assert seen["segment_kwargs"]["split_mode"] == "aggressive"
    assert seen["pnm_kwargs"]["segmentation_profile"] == "legacy"
    assert seen["pnm_kwargs"]["split_mode"] == "aggressive"
