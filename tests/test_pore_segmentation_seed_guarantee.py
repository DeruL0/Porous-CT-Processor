import numpy as np

from processors.pore_segmentation import PoreSegmentationConfig, segment_pores_from_mask


def test_each_component_gets_at_least_one_seed():
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[10:13, 10:13, 10:13] = True
    mask[10:13, 10:13, 16:19] = True

    cfg = PoreSegmentationConfig(
        profile="dual_adaptive",
        split_mode="balanced",
        base_min_peak_distance=6,
        force_seed_per_component=True,
    )
    result = segment_pores_from_mask(mask, spacing=(1.0, 1.0, 1.0), config=cfg)

    assert result.debug["num_components"] == 2
    assert result.debug["num_markers"] >= 2
    assert result.num_pores >= 2
