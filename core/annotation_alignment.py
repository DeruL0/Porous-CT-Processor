"""Helpers for aligning predicted label volumes with simulation annotations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


_GT_PERMUTATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


@dataclass(frozen=True)
class ShapeAlignment:
    """Axis permutation + center-crop windows for shape alignment."""

    gt_perm: Tuple[int, int, int]
    pred_slices: Tuple[slice, slice, slice]
    gt_slices: Tuple[slice, slice, slice]
    overlap_shape: Tuple[int, int, int]
    overlap_ratio: float

    def to_dict(self) -> dict:
        return {
            "gt_perm": list(self.gt_perm),
            "pred_slices": [[s.start, s.stop] for s in self.pred_slices],
            "gt_slices": [[s.start, s.stop] for s in self.gt_slices],
            "overlap_shape": list(self.overlap_shape),
            "overlap_ratio": float(self.overlap_ratio),
        }


def _normalize_shape(shape: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    if len(shape) != 3:
        return None
    out = tuple(int(v) for v in shape)
    if any(v <= 0 for v in out):
        return None
    return out


def _center_crop_slices(shape: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    slices = []
    for dim, size in zip(shape, target):
        start = max(0, (dim - size) // 2)
        slices.append(slice(start, start + size))
    return tuple(slices)  # type: ignore[return-value]


def infer_shape_alignment(
    pred_shape: Sequence[int],
    gt_shape: Sequence[int],
    min_overlap_ratio: float = 0.85,
) -> Optional[ShapeAlignment]:
    """
    Infer deterministic alignment from predicted shape to GT shape.

    The GT volume may be axis-permuted and slightly larger/smaller.
    Alignment chooses:
    1) max overlap ratio,
    2) min crop amount,
    3) stable permutation order.
    """
    pred = _normalize_shape(pred_shape)
    gt = _normalize_shape(gt_shape)
    if pred is None or gt is None:
        return None

    best_key = None
    best_alignment: Optional[ShapeAlignment] = None
    pred_voxels = math.prod(pred)

    for perm_rank, perm in enumerate(_GT_PERMUTATIONS):
        gt_perm_shape = (gt[perm[0]], gt[perm[1]], gt[perm[2]])
        overlap = tuple(min(pred[i], gt_perm_shape[i]) for i in range(3))
        overlap_voxels = math.prod(overlap)
        if overlap_voxels <= 0:
            continue

        envelope = tuple(max(pred[i], gt_perm_shape[i]) for i in range(3))
        envelope_voxels = math.prod(envelope)
        overlap_ratio = float(overlap_voxels / envelope_voxels) if envelope_voxels > 0 else 0.0

        gt_voxels = math.prod(gt_perm_shape)
        crop_penalty = (pred_voxels - overlap_voxels) + (gt_voxels - overlap_voxels)

        pred_slices = _center_crop_slices(pred, overlap)
        gt_slices = _center_crop_slices(gt_perm_shape, overlap)
        alignment = ShapeAlignment(
            gt_perm=perm,
            pred_slices=pred_slices,
            gt_slices=gt_slices,
            overlap_shape=overlap,
            overlap_ratio=overlap_ratio,
        )

        key = (-overlap_ratio, crop_penalty, perm_rank)
        if best_key is None or key < best_key:
            best_key = key
            best_alignment = alignment

    if best_alignment is None:
        return None
    if best_alignment.overlap_ratio < float(min_overlap_ratio):
        return None
    return best_alignment


def align_pred_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    alignment: ShapeAlignment,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an inferred alignment and return view-like arrays."""
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    gt_permuted = np.transpose(gt_arr, alignment.gt_perm)
    pred_view = pred_arr[alignment.pred_slices]
    gt_view = gt_permuted[alignment.gt_slices]
    return pred_view, gt_view

