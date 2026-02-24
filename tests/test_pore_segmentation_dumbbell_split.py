import numpy as np

from processors.pore_segmentation import PoreSegmentationConfig, segment_pores_from_mask


def test_balanced_mode_splits_dumbbell_component():
    shape = (48, 48, 48)
    z, y, x = np.indices(shape)

    big = (z - 24) ** 2 + (y - 24) ** 2 + (x - 15) ** 2 <= 8 ** 2
    small = (z - 24) ** 2 + (y - 24) ** 2 + (x - 34) ** 2 <= 4 ** 2
    throat = ((z - 24) ** 2 + (y - 24) ** 2 <= 2 ** 2) & (x >= 15) & (x <= 34)
    mask = big | small | throat

    cfg = PoreSegmentationConfig(
        profile="dual_adaptive",
        split_mode="balanced",
        neck_erosion_iters=1,
    )
    result = segment_pores_from_mask(mask, spacing=(1.0, 1.0, 1.0), config=cfg)

    label_big = int(result.labels[24, 24, 15])
    label_small = int(result.labels[24, 24, 34])

    assert label_big > 0
    assert label_small > 0
    assert label_big != label_small


def test_conservative_mode_keeps_dumbbell_component_single():
    shape = (48, 48, 48)
    z, y, x = np.indices(shape)

    big = (z - 24) ** 2 + (y - 24) ** 2 + (x - 15) ** 2 <= 8 ** 2
    small = (z - 24) ** 2 + (y - 24) ** 2 + (x - 34) ** 2 <= 4 ** 2
    throat = ((z - 24) ** 2 + (y - 24) ** 2 <= 2 ** 2) & (x >= 15) & (x <= 34)
    mask = big | small | throat

    cfg = PoreSegmentationConfig(
        profile="dual_adaptive",
        split_mode="conservative",
        neck_erosion_iters=1,
    )
    result = segment_pores_from_mask(mask, spacing=(1.0, 1.0, 1.0), config=cfg)

    label_big = int(result.labels[24, 24, 15])
    label_small = int(result.labels[24, 24, 34])

    assert label_big > 0
    assert label_small > 0
    assert label_big == label_small
