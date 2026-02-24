import numpy as np

from processors.pore_segmentation import PoreSegmentationConfig, segment_pores_from_raw
from processors.utils import binary_fill_holes


def test_legacy_profile_keeps_legacy_mask_definition():
    raw = np.full((32, 32, 32), 1000.0, dtype=np.float32)
    raw[10:18, 10:18, 10:18] = 0.0
    raw[13:15, 13:15, 13:15] = 1000.0

    threshold = 500.0
    legacy_mask = binary_fill_holes(raw > threshold) ^ (raw > threshold)

    cfg = PoreSegmentationConfig(profile="legacy", split_mode="balanced")
    result = segment_pores_from_raw(raw, threshold=threshold, spacing=(1.0, 1.0, 1.0), config=cfg)

    assert np.array_equal(result.pores_mask, legacy_mask)
    assert result.debug["profile"] == "legacy"
