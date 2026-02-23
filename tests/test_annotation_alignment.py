import numpy as np
import pytest

from core.annotation_alignment import align_pred_gt, infer_shape_alignment


def test_infer_shape_alignment_detects_axis_permutation():
    alignment = infer_shape_alignment((20, 30, 40), (40, 30, 20))
    assert alignment is not None
    assert alignment.gt_perm == (2, 1, 0)
    assert alignment.overlap_shape == (20, 30, 40)
    assert alignment.overlap_ratio == pytest.approx(1.0)


def test_align_pred_gt_returns_consistent_views():
    pred = np.zeros((6, 8, 10), dtype=np.int32)
    pred[1:4, 2:6, 3:8] = 7
    gt = np.transpose(pred, (2, 1, 0))

    alignment = infer_shape_alignment(pred.shape, gt.shape)
    assert alignment is not None

    pred_view, gt_view = align_pred_gt(pred, gt, alignment)
    np.testing.assert_array_equal(pred_view, gt_view)


def test_infer_shape_alignment_rejects_low_overlap():
    alignment = infer_shape_alignment((100, 100, 100), (30, 30, 30), min_overlap_ratio=0.85)
    assert alignment is None

