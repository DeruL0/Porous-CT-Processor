"""
Core matching and tracking algorithms extracted from pnm_tracker.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.coordinates import world_delta_xyz_to_voxel_delta_zyx
from core.time_series import PNMSnapshot, PoreStatus, PoreTrackingResult
from processors.tracking_utils import (
    ConstantAccelerationKalman3D,
    bounded_volume_penalty,
    extract_shifted_overlap_region,
)

INVALID_COST = 1e6

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    linear_sum_assignment = None

HAS_LAPJV = False
_LAPJV_SOLVER = None
try:
    from lapjv import lapjv as _LAPJV_SOLVER

    HAS_LAPJV = True
except ImportError:
    try:
        from lap import lapjv as _LAPJV_SOLVER

        HAS_LAPJV = True
    except ImportError:
        HAS_LAPJV = False
        _LAPJV_SOLVER = None


def compute_pair_cost(
    iou: float,
    center_dist: float,
    reference_volume: float,
    current_volume: float,
    dice_local: float,
    cost_weights: Tuple[float, float, float, float],
    center_gate_radius: float,
    volume_cost_mode: str = "symdiff",
    volume_cost_sigma: float = 1.0,
) -> float:
    """Compute TGGA pairwise matching cost."""
    w_iou, w_center, w_volume, w_dice = cost_weights
    norm_center_dist = center_dist / max(center_gate_radius, 1e-8)
    norm_center_dist = float(np.clip(norm_center_dist, 0.0, 1.0))
    volume_term = bounded_volume_penalty(
        reference_volume=reference_volume,
        current_volume=current_volume,
        mode=volume_cost_mode,
        gaussian_sigma=volume_cost_sigma,
    )
    dice_term = 1.0 - float(np.clip(dice_local, 0.0, 1.0))
    return (
        w_iou * (1.0 - float(np.clip(iou, 0.0, 1.0)))
        + w_center * norm_center_dist
        + w_volume * volume_term
        + w_dice * dice_term
    )


def _compute_overlap_metrics(
    local_ref_mask: np.ndarray,
    current_region: np.ndarray,
    label_id: int,
) -> Tuple[float, float, float, int]:
    """Compute IoU, local Dice, and local candidate volume in cropped bbox."""
    candidate_mask = current_region == int(label_id)
    intersection = int(np.sum(local_ref_mask & candidate_mask))
    ref_volume = int(np.sum(local_ref_mask))
    cand_volume = int(np.sum(candidate_mask))
    union = ref_volume + cand_volume - intersection
    iou = (intersection / union) if union > 0 else 0.0
    dice = (2.0 * intersection / (ref_volume + cand_volume)) if (ref_volume + cand_volume) > 0 else 0.0
    return float(iou), float(dice), float(cand_volume), int(intersection)


def build_candidates(
    reference_snapshot: PNMSnapshot,
    current_snapshot: PNMSnapshot,
    reference_masks: Dict[int, Dict],
    current_regions: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    predicted_centers: Dict[int, np.ndarray],
    cost_weights: Tuple[float, float, float, float],
    gate_center_radius_factor: float,
    gate_volume_ratio_min: float,
    gate_volume_ratio_max: float,
    gate_iou_min: float = 0.02,
    gate_volume_ratio_min_floor: float = 1e-4,
    small_pore_volume_voxels: float = 64.0,
    soft_gate_min_intersection_voxels: int = 3,
    soft_gate_cost_penalty: float = 0.08,
    volume_cost_mode: str = "symdiff",
    volume_cost_sigma: float = 1.0,
) -> Dict[str, object]:
    """
    Build TGGA candidate graph and dense cost matrix.

    Returns:
        {
            "cost_matrix": np.ndarray (N_ref x N_curr),
            "pair_metrics": {(ref_idx, curr_idx): dict},
            "row_geom_candidate": np.ndarray(bool),
            "row_valid_candidate": np.ndarray(bool),
        }
    """
    ref_ids = np.asarray(reference_snapshot.pore_ids, dtype=np.int32)
    curr_ids = (
        np.asarray(current_snapshot.pore_ids, dtype=np.int32)
        if current_snapshot.pore_ids is not None
        else np.array([], dtype=np.int32)
    )
    curr_centers = (
        np.asarray(current_snapshot.pore_centers, dtype=np.float64)
        if current_snapshot.pore_centers is not None
        else np.zeros((len(curr_ids), 3), dtype=np.float64)
    )
    curr_volumes = (
        np.asarray(current_snapshot.pore_volumes, dtype=np.float64)
        if current_snapshot.pore_volumes is not None
        else np.zeros((len(curr_ids),), dtype=np.float64)
    )

    n_ref = len(ref_ids)
    n_curr = len(curr_ids)
    cost_matrix = np.full((n_ref, n_curr), INVALID_COST, dtype=np.float64)
    pair_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    row_geom_candidate = np.zeros(n_ref, dtype=bool)
    row_valid_candidate = np.zeros(n_ref, dtype=bool)

    if n_ref == 0 or n_curr == 0:
        return {
            "cost_matrix": cost_matrix,
            "pair_metrics": pair_metrics,
            "row_geom_candidate": row_geom_candidate,
            "row_valid_candidate": row_valid_candidate,
        }

    ref_centers = np.asarray(reference_snapshot.pore_centers, dtype=np.float64)
    ref_radii = np.asarray(reference_snapshot.pore_radii, dtype=np.float64)
    ref_volumes = np.asarray(reference_snapshot.pore_volumes, dtype=np.float64)
    effective_ratio_min = min(float(gate_volume_ratio_min), float(gate_volume_ratio_min_floor))

    for ref_idx, ref_id in enumerate(ref_ids):
        ref_id = int(ref_id)
        mask_data = reference_masks.get(ref_id)
        if mask_data is None:
            continue

        predicted = np.asarray(predicted_centers.get(ref_id, ref_centers[ref_idx]), dtype=np.float64)
        ref_radius = float(max(ref_radii[ref_idx], 1e-6))
        center_gate_radius = float(max(gate_center_radius_factor * ref_radius, 1e-6))

        dists = np.linalg.norm(curr_centers - predicted, axis=1)
        ref_volume = float(ref_volumes[ref_idx])
        if ref_volume > 0:
            volume_ratios = curr_volumes / ref_volume
        else:
            volume_ratios = np.full_like(curr_volumes, np.inf, dtype=np.float64)

        geometric_gate = (
            (dists <= center_gate_radius)
            & (volume_ratios >= effective_ratio_min)
            & (volume_ratios <= gate_volume_ratio_max)
        )
        if np.any(geometric_gate):
            row_geom_candidate[ref_idx] = True
        else:
            continue

        mins, _maxs = mask_data["bbox"]
        local_ref_mask = mask_data["mask"]
        delta_world_xyz = predicted - ref_centers[ref_idx]
        shift_zyx = world_delta_xyz_to_voxel_delta_zyx(
            delta_xyz=(float(delta_world_xyz[0]), float(delta_world_xyz[1]), float(delta_world_xyz[2])),
            spacing_xyz=spacing_xyz,
        )
        shifted_ref_mask, shifted_current_region = extract_shifted_overlap_region(
            current_regions=current_regions,
            local_reference_mask=local_ref_mask,
            bbox_mins=np.asarray(mins, dtype=np.int64),
            shift_zyx=shift_zyx,
        )
        if shifted_ref_mask.size == 0:
            continue

        gated_curr_indices = np.where(geometric_gate)[0]
        for curr_idx in gated_curr_indices:
            curr_id = int(curr_ids[curr_idx])
            iou, dice_local, _local_volume, intersection_voxels = _compute_overlap_metrics(
                shifted_ref_mask,
                shifted_current_region,
                curr_id,
            )
            volume_ratio = float(volume_ratios[curr_idx])
            adaptive_iou_gate = min(float(gate_iou_min), max(1e-4, 0.5 * min(volume_ratio, 1.0)))
            soft_gate_used = False
            if iou < adaptive_iou_gate:
                if ref_volume <= float(small_pore_volume_voxels) and intersection_voxels >= int(
                    max(1, soft_gate_min_intersection_voxels)
                ):
                    soft_gate_used = True
                else:
                    continue

            row_valid_candidate[ref_idx] = True
            center_dist = float(dists[curr_idx])
            curr_volume = float(curr_volumes[curr_idx])
            cost = compute_pair_cost(
                iou=iou,
                center_dist=center_dist,
                reference_volume=ref_volume,
                current_volume=curr_volume,
                dice_local=dice_local,
                cost_weights=cost_weights,
                center_gate_radius=center_gate_radius,
                volume_cost_mode=volume_cost_mode,
                volume_cost_sigma=volume_cost_sigma,
            )
            if soft_gate_used:
                cost += float(max(0.0, soft_gate_cost_penalty))

            cost_matrix[ref_idx, curr_idx] = min(cost_matrix[ref_idx, curr_idx], cost)
            pair_metrics[(ref_idx, curr_idx)] = {
                "iou": iou,
                "dice_local": dice_local,
                "center_dist": center_dist,
                "norm_center_dist": center_dist / center_gate_radius,
                "volume_ratio": volume_ratio,
                "intersection_voxels": int(intersection_voxels),
                "soft_gate_used": bool(soft_gate_used),
                "volume_penalty": bounded_volume_penalty(
                    reference_volume=ref_volume,
                    current_volume=curr_volume,
                    mode=volume_cost_mode,
                    gaussian_sigma=volume_cost_sigma,
                ),
                "cost": cost,
            }
            if soft_gate_used:
                continue

    return {
        "cost_matrix": cost_matrix,
        "pair_metrics": pair_metrics,
        "row_geom_candidate": row_geom_candidate,
        "row_valid_candidate": row_valid_candidate,
    }


def _solve_with_lapjv(
    cost_matrix: np.ndarray,
    invalid_cost: float = INVALID_COST,
) -> List[Tuple[int, int, float]]:
    """Solve assignment with lapjv/lap package, handling rectangular matrices."""
    if not HAS_LAPJV or _LAPJV_SOLVER is None:
        raise RuntimeError("lapjv solver requested but lapjv/lap is unavailable")

    n_ref, n_curr = cost_matrix.shape
    if n_ref == 0 or n_curr == 0:
        return []

    n = max(n_ref, n_curr)
    padded = np.full((n, n), float(invalid_cost), dtype=np.float64)
    padded[:n_ref, :n_curr] = cost_matrix

    result = _LAPJV_SOLVER(padded)
    row_to_col: Optional[np.ndarray] = None

    if isinstance(result, tuple):
        if len(result) == 3 and np.isscalar(result[0]):
            row_to_col = np.asarray(result[1], dtype=np.int64)
        elif len(result) >= 2:
            first = np.asarray(result[0])
            second = np.asarray(result[1])
            if first.ndim == 1 and first.shape[0] == n:
                row_to_col = first.astype(np.int64)
            elif second.ndim == 1 and second.shape[0] == n:
                row_to_col = second.astype(np.int64)
    if row_to_col is None:
        raise RuntimeError("Unexpected lapjv return signature")

    matches: List[Tuple[int, int, float]] = []
    for ref_idx in range(n_ref):
        curr_idx = int(row_to_col[ref_idx])
        if curr_idx < 0 or curr_idx >= n_curr:
            continue
        cost = float(cost_matrix[ref_idx, curr_idx])
        if cost < invalid_cost * 0.5:
            matches.append((ref_idx, curr_idx, cost))
    return matches


def _solve_with_scipy(
    cost_matrix: np.ndarray,
    invalid_cost: float = INVALID_COST,
) -> List[Tuple[int, int, float]]:
    """Solve assignment with scipy's Hungarian implementation."""
    if not HAS_SCIPY or linear_sum_assignment is None:
        raise RuntimeError("scipy solver requested but scipy is unavailable")

    n_ref, n_curr = cost_matrix.shape
    if n_ref == 0 or n_curr == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int, float]] = []
    for ref_idx, curr_idx in zip(row_ind, col_ind):
        cost = float(cost_matrix[ref_idx, curr_idx])
        if cost < invalid_cost * 0.5:
            matches.append((int(ref_idx), int(curr_idx), cost))
    return matches


def solve_global_assignment(
    cost_matrix: np.ndarray,
    assign_solver: str = "lapjv",
    invalid_cost: float = INVALID_COST,
) -> Tuple[List[Tuple[int, int, float]], str]:
    """
    Solve one-to-one global assignment on a dense cost matrix.

    Returns:
        (matches, solver_used), where matches are (ref_idx, curr_idx, cost).
    """
    if cost_matrix.size == 0:
        return [], assign_solver

    solver = (assign_solver or "lapjv").lower()
    if solver == "lapjv":
        if HAS_LAPJV:
            try:
                return _solve_with_lapjv(cost_matrix, invalid_cost=invalid_cost), "lapjv"
            except Exception:
                if not HAS_SCIPY:
                    raise
        if HAS_SCIPY:
            return _solve_with_scipy(cost_matrix, invalid_cost=invalid_cost), "scipy"
        raise RuntimeError("No assignment solver available (lapjv and scipy are both unavailable)")

    if solver == "scipy":
        if HAS_SCIPY:
            return _solve_with_scipy(cost_matrix, invalid_cost=invalid_cost), "scipy"
        if HAS_LAPJV:
            return _solve_with_lapjv(cost_matrix, invalid_cost=invalid_cost), "lapjv"
        raise RuntimeError("No assignment solver available (scipy and lapjv are both unavailable)")

    raise ValueError(f"Unsupported assignment solver: {assign_solver}")


def update_tracks_with_hysteresis(
    tracking: PoreTrackingResult,
    reference_snapshot: PNMSnapshot,
    current_center_map: Dict[int, np.ndarray],
    match_results: Dict[int, Dict[str, object]],
    max_misses: int,
    iou_threshold: float,
    compression_threshold: float,
    center_smoothing: float,
    predicted_centers: Optional[Dict[int, np.ndarray]] = None,
    track_predictors: Optional[Dict[int, ConstantAccelerationKalman3D]] = None,
    closed_reason_token: str = "closed_by_compression",
    kalman_brake_velocity_decay: float = 0.75,
    kalman_brake_acceleration_decay: float = 0.35,
    kalman_freeze_after_misses: int = 3,
) -> Dict[int, int]:
    """Update time-series histories with consecutive-miss hysteresis."""
    id_map: Dict[int, int] = {}
    ref_ids = np.asarray(reference_snapshot.pore_ids, dtype=np.int32)
    ref_centers = np.asarray(reference_snapshot.pore_centers, dtype=np.float64)
    predicted_center_map = predicted_centers or {}
    predictor_map = track_predictors or {}

    for i, ref_id_raw in enumerate(ref_ids):
        ref_id = int(ref_id_raw)
        info = match_results.get(ref_id, {"matched_id": -1, "reason": "unmatched"})
        predictor = predictor_map.get(ref_id)

        history = tracking.center_history.setdefault(ref_id, [])
        prev_center = np.array(history[-1], dtype=np.float64) if history else ref_centers[i]

        tracking.volume_history.setdefault(ref_id, [])
        tracking.status_history.setdefault(ref_id, [])
        tracking.iou_history.setdefault(ref_id, [])
        tracking.match_confidence.setdefault(ref_id, [])
        tracking.unmatched_reason.setdefault(ref_id, [])
        tracking.miss_count.setdefault(ref_id, 0)

        matched_id = int(info.get("matched_id", -1))
        if matched_id > 0:
            iou = float(info.get("iou", 0.0))
            current_volume = float(info.get("current_volume", 0.0))
            volume_ratio = float(info.get("volume_ratio", 0.0))
            cost = float(info.get("cost", 1.0))

            tracking.miss_count[ref_id] = 0
            if iou >= iou_threshold and volume_ratio >= compression_threshold:
                status = PoreStatus.ACTIVE
            else:
                status = PoreStatus.COMPRESSED

            confidence = float(np.clip(1.0 - cost, 0.0, 1.0))
            tracking.volume_history[ref_id].append(current_volume)
            tracking.status_history[ref_id].append(status)
            tracking.iou_history[ref_id].append(iou)
            tracking.match_confidence[ref_id].append(confidence)
            tracking.unmatched_reason[ref_id].append("matched")
            id_map[ref_id] = matched_id

            if matched_id in current_center_map:
                curr_center = np.asarray(current_center_map[matched_id], dtype=np.float64)
                if predictor is not None:
                    filtered_center = predictor.update(curr_center)
                    history.append(filtered_center.tolist())
                else:
                    smoothed = prev_center + center_smoothing * (curr_center - prev_center)
                    history.append(smoothed.tolist())
            else:
                if predictor is not None:
                    history.append(predictor.current_position().tolist())
                else:
                    history.append(prev_center.tolist())
        else:
            misses = int(tracking.miss_count.get(ref_id, 0)) + 1
            tracking.miss_count[ref_id] = misses
            reason = str(info.get("reason", "unmatched"))
            if reason == closed_reason_token:
                status = PoreStatus.COMPRESSED
                carry_volume = 0.0
            else:
                status = PoreStatus.ACTIVE if misses <= int(max_misses) else PoreStatus.COMPRESSED
                prev_volume = tracking.volume_history[ref_id][-1] if tracking.volume_history[ref_id] else 0.0
                carry_volume = float(prev_volume if status == PoreStatus.ACTIVE else 0.0)
            if predictor is not None:
                predicted_center = predictor.current_position()
            else:
                predicted_center = np.asarray(predicted_center_map.get(ref_id, prev_center), dtype=np.float64)

            tracking.volume_history[ref_id].append(carry_volume)
            tracking.status_history[ref_id].append(status)
            tracking.iou_history[ref_id].append(0.0)
            tracking.match_confidence[ref_id].append(0.0)
            tracking.unmatched_reason[ref_id].append(reason)
            id_map[ref_id] = -1
            if predictor is not None:
                predictor.apply_brake(
                    miss_count=misses,
                    velocity_decay=kalman_brake_velocity_decay,
                    acceleration_decay=kalman_brake_acceleration_decay,
                    freeze_after_misses=kalman_freeze_after_misses,
                )
            history.append(predicted_center.tolist())

    return id_map
