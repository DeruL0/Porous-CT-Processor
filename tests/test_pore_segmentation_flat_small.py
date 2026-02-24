import numpy as np

from processors.pore_segmentation import PoreSegmentationConfig, segment_pores_from_raw


def test_flat_and_small_pores_are_detected():
    raw = np.full((40, 40, 40), 1000.0, dtype=np.float32)
    raw[15:16, 10:20, 10:20] = 0.0
    raw[25:27, 25:27, 25:27] = 0.0

    expected = np.zeros_like(raw, dtype=bool)
    expected[15:16, 10:20, 10:20] = True
    expected[25:27, 25:27, 25:27] = True

    cfg = PoreSegmentationConfig(profile="dual_adaptive", split_mode="balanced")
    result = segment_pores_from_raw(raw, threshold=500.0, spacing=(1.0, 1.0, 1.0), config=cfg)

    overlap = int(np.count_nonzero(result.pores_mask & expected))
    assert overlap >= int(expected.sum() * 0.95)
    assert result.num_pores >= 2
