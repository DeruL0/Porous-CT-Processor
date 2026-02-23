import numpy as np

from core import VolumeData
from data.manager import ScientificDataManager


def test_clip_volume_inplace_clips_and_updates_metadata_once():
    raw = np.array(
        [
            [[-500.0, -100.0], [0.0, 300.0]],
            [[600.0, 1200.0], [-350.0, 50.0]],
        ],
        dtype=np.float32,
    )
    volume = VolumeData(raw_data=raw.copy(), metadata={"Type": "CT"})
    raw_ref = volume.raw_data

    ScientificDataManager.clip_volume_inplace(volume, min_val=-200.0, max_val=200.0)
    assert volume.raw_data is raw_ref
    assert float(volume.raw_data.min()) >= -200.0
    assert float(volume.raw_data.max()) <= 200.0
    assert volume.metadata["ClipRange"] == "[-200, 200]"
    assert volume.metadata["Type"].count("(Clipped)") == 1

    ScientificDataManager.clip_volume_inplace(volume, min_val=-150.0, max_val=150.0)
    assert volume.metadata["Type"].count("(Clipped)") == 1


def test_invert_volume_inplace_returns_offset_and_updates_data():
    raw = np.array(
        [
            [[-5.0, 0.0], [10.0, 20.0]],
            [[30.0, -10.0], [5.0, 1.0]],
        ],
        dtype=np.float32,
    )
    original = raw.copy()
    volume = VolumeData(raw_data=raw, metadata={"Type": "CT"})

    data_min, data_max, offset = ScientificDataManager.invert_volume_inplace(volume)
    expected_offset = float(original.min() + original.max())
    expected = expected_offset - original

    assert data_min == float(original.min())
    assert data_max == float(original.max())
    assert offset == expected_offset
    np.testing.assert_allclose(volume.raw_data, expected)
    assert volume.metadata["Type"].count("(Inverted)") == 1

    ScientificDataManager.invert_volume_inplace(volume)
    assert volume.metadata["Type"].count("(Inverted)") == 1
