"""
Pore Network Model Tracker for 4D CT analysis.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.annotation_alignment import align_pred_gt, infer_shape_alignment
from core.coordinates import (
    voxel_delta_zyx_to_world_delta_xyz,
    world_delta_xyz_to_voxel_delta_zyx,
    world_xyz_to_voxel_zyx,
)
from core.time_series import PNMSnapshot, PoreStatus, PoreTrackingResult, TimeSeriesPNM
from config import (
    TRACKING_ASSIGN_SOLVER,
    TRACKING_CENTER_SMOOTHING,
    TRACKING_CLOSURE_MIN_VOLUME_VOXELS,
    TRACKING_CLOSURE_STRAIN_THRESHOLD,
    TRACKING_CLOSURE_VOLUME_RATIO_THRESHOLD,
    TRACKING_COMPRESSION_THRESHOLD,
    TRACKING_COST_WEIGHTS,
    TRACKING_ENABLE_MACRO_REGISTRATION,
    TRACKING_GATE_CENTER_RADIUS_FACTOR,
    TRACKING_GATE_IOU_MIN,
    TRACKING_GATE_VOLUME_RATIO_MIN_FLOOR,
    TRACKING_GATE_VOLUME_RATIO_MAX,
    TRACKING_GATE_VOLUME_RATIO_MIN,
    TRACKING_GPU_MIN_PORES,
    TRACKING_IOU_THRESHOLD,
    TRACKING_KALMAN_BRAKE_ACCEL_DECAY,
    TRACKING_KALMAN_BRAKE_VELOCITY_DECAY,
    TRACKING_KALMAN_FREEZE_AFTER_MISSES,
    TRACKING_KALMAN_MEASUREMENT_NOISE,
    TRACKING_KALMAN_PROCESS_NOISE,
    TRACKING_MACRO_REG_SMOOTHING_SIGMA,
    TRACKING_MACRO_REG_USE_GPU,
    TRACKING_MACRO_REG_GPU_MIN_MB,
    TRACKING_MACRO_REG_UPSAMPLE_FACTOR,
    TRACKING_MATCH_MODE,
    TRACKING_MAX_MISSES,
    TRACKING_NOVEL_MIN_PERSISTENCE,
    TRACKING_NOVEL_MIN_VOLUME_VOXELS,
    TRACKING_SMALL_PORE_VOLUME_VOXELS,
    TRACKING_SOFT_GATE_COST_PENALTY,
    TRACKING_SOFT_GATE_MIN_INTERSECTION_VOXELS,
    TRACKING_USE_BATCH,
    TRACKING_USE_GPU,
    TRACKING_USE_HUNGARIAN,
    TRACKING_VOLUME_COST_GAUSSIAN_SIGMA,
    TRACKING_VOLUME_COST_MODE,
)
from processors.tracking_utils import (
    MacroRegistrationResult,
    ConstantAccelerationKalman3D,
    bounded_volume_penalty,
    estimate_macro_registration,
    extract_shifted_overlap_region,
    should_mark_closed_by_compression,
)

INVALID_COST = 1e6

# Try to import GPU backend
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Try to import scipy assignment solver
try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional Jonker-Volgenant solvers
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
    """
    Compute TGGA pairwise matching cost.

    cost = w1*(1-IoU) + w2*norm_center_dist + w3*C_vol + w4*(1-dice_local)
    where C_vol is bounded to [0, 1].
    """
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
    """Compute IoU, local Dice, and local candidate volume for a label in a cropped bbox."""
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
                # Keep the soft candidate but do not skip stronger candidates.
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
    """Solve assignment with lapjv/lap package, handling rectangular matrices by padding."""
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
    if not HAS_SCIPY:
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


class PNMTracker:
    """
    Tracks pore network changes across 4D CT time series.

    This tracker maintains correspondence between pores in a reference
    snapshot (t=0) and all subsequent timepoints.
    """

    def __init__(
        self,
        iou_threshold: float = None,
        compression_threshold: float = None,
        use_gpu: bool = None,
        use_batch: bool = None,
        use_hungarian: bool = None,
        match_mode: str = None,
        assign_solver: str = None,
        cost_weights: Tuple[float, float, float, float] = None,
        max_misses: int = None,
        gating_params: Dict[str, float] = None,
    ):
        self.iou_threshold = iou_threshold if iou_threshold is not None else TRACKING_IOU_THRESHOLD
        self.compression_threshold = (
            compression_threshold if compression_threshold is not None else TRACKING_COMPRESSION_THRESHOLD
        )

        self.use_gpu = use_gpu if use_gpu is not None else (TRACKING_USE_GPU and HAS_GPU)
        self.use_batch = use_batch if use_batch is not None else TRACKING_USE_BATCH
        self.center_smoothing = float(TRACKING_CENTER_SMOOTHING)

        resolved_match_mode = (match_mode or TRACKING_MATCH_MODE).lower()
        if use_hungarian is not None:
            warnings.warn(
                "use_hungarian is deprecated. Use match_mode='global_iou_legacy' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if use_hungarian:
                resolved_match_mode = "global_iou_legacy"
        else:
            if TRACKING_USE_HUNGARIAN:
                resolved_match_mode = "global_iou_legacy"

        if resolved_match_mode not in {"temporal_global", "legacy_greedy", "global_iou_legacy"}:
            warnings.warn(
                f"Unknown match_mode='{resolved_match_mode}', fallback to temporal_global",
                RuntimeWarning,
            )
            resolved_match_mode = "temporal_global"
        self.match_mode = resolved_match_mode

        self.assign_solver = (assign_solver or TRACKING_ASSIGN_SOLVER).lower()
        if self.assign_solver not in {"lapjv", "scipy"}:
            warnings.warn(
                f"Unknown assign_solver='{self.assign_solver}', fallback to lapjv",
                RuntimeWarning,
            )
            self.assign_solver = "lapjv"
        if self.assign_solver == "lapjv" and not HAS_LAPJV and HAS_SCIPY:
            warnings.warn("lapjv requested but unavailable, falling back to scipy solver")
            self.assign_solver = "scipy"
        elif self.assign_solver == "scipy" and not HAS_SCIPY and HAS_LAPJV:
            warnings.warn("scipy solver requested but unavailable, falling back to lapjv")
            self.assign_solver = "lapjv"

        if self.assign_solver == "lapjv" and not HAS_LAPJV and not HAS_SCIPY:
            raise RuntimeError("No assignment solver available (lapjv/scipy both unavailable)")
        if self.assign_solver == "scipy" and not HAS_SCIPY and not HAS_LAPJV:
            raise RuntimeError("No assignment solver available (scipy/lapjv both unavailable)")

        weights = cost_weights if cost_weights is not None else TRACKING_COST_WEIGHTS
        if len(weights) != 4:
            warnings.warn("cost_weights must contain 4 values. Falling back to default weights.")
            weights = TRACKING_COST_WEIGHTS
        self.cost_weights = tuple(float(w) for w in weights)

        self.max_misses = int(max_misses if max_misses is not None else TRACKING_MAX_MISSES)
        gates = gating_params or {}
        self.gate_center_radius_factor = float(
            gates.get("center_radius_factor", TRACKING_GATE_CENTER_RADIUS_FACTOR)
        )
        self.gate_volume_ratio_min = float(gates.get("volume_ratio_min", TRACKING_GATE_VOLUME_RATIO_MIN))
        self.gate_volume_ratio_max = float(gates.get("volume_ratio_max", TRACKING_GATE_VOLUME_RATIO_MAX))
        self.gate_iou_min = float(gates.get("iou_min", TRACKING_GATE_IOU_MIN))
        self.gate_volume_ratio_min_floor = float(
            gates.get("volume_ratio_min_floor", TRACKING_GATE_VOLUME_RATIO_MIN_FLOOR)
        )
        self.small_pore_volume_voxels = float(
            gates.get("small_pore_volume_voxels", TRACKING_SMALL_PORE_VOLUME_VOXELS)
        )
        self.soft_gate_min_intersection_voxels = int(
            gates.get("soft_gate_min_intersection_voxels", TRACKING_SOFT_GATE_MIN_INTERSECTION_VOXELS)
        )
        self.soft_gate_cost_penalty = float(
            gates.get("soft_gate_cost_penalty", TRACKING_SOFT_GATE_COST_PENALTY)
        )
        self.novel_min_volume_voxels = float(
            gates.get("novel_min_volume_voxels", TRACKING_NOVEL_MIN_VOLUME_VOXELS)
        )
        self.novel_min_persistence = int(
            gates.get("novel_min_persistence", TRACKING_NOVEL_MIN_PERSISTENCE)
        )

        self.volume_cost_mode = str(gates.get("volume_cost_mode", TRACKING_VOLUME_COST_MODE))
        self.volume_cost_sigma = float(
            gates.get("volume_cost_sigma", TRACKING_VOLUME_COST_GAUSSIAN_SIGMA)
        )

        self.enable_macro_registration = bool(
            gates.get("enable_macro_registration", TRACKING_ENABLE_MACRO_REGISTRATION)
        )
        self.macro_reg_smoothing_sigma = float(
            gates.get("macro_reg_smoothing_sigma", TRACKING_MACRO_REG_SMOOTHING_SIGMA)
        )
        self.macro_reg_upsample_factor = int(
            gates.get("macro_reg_upsample_factor", TRACKING_MACRO_REG_UPSAMPLE_FACTOR)
        )
        self.macro_reg_use_gpu = bool(
            gates.get("macro_reg_use_gpu", TRACKING_MACRO_REG_USE_GPU)
        )
        self.macro_reg_gpu_min_mb = float(
            gates.get("macro_reg_gpu_min_mb", TRACKING_MACRO_REG_GPU_MIN_MB)
        )

        self.kalman_process_noise = float(
            gates.get("kalman_process_noise", TRACKING_KALMAN_PROCESS_NOISE)
        )
        self.kalman_measurement_noise = float(
            gates.get("kalman_measurement_noise", TRACKING_KALMAN_MEASUREMENT_NOISE)
        )
        self.kalman_brake_velocity_decay = float(
            gates.get("kalman_brake_velocity_decay", TRACKING_KALMAN_BRAKE_VELOCITY_DECAY)
        )
        self.kalman_brake_acceleration_decay = float(
            gates.get("kalman_brake_acceleration_decay", TRACKING_KALMAN_BRAKE_ACCEL_DECAY)
        )
        self.kalman_freeze_after_misses = int(
            gates.get("kalman_freeze_after_misses", TRACKING_KALMAN_FREEZE_AFTER_MISSES)
        )

        self.closure_volume_ratio_threshold = float(
            gates.get("closure_volume_ratio_threshold", TRACKING_CLOSURE_VOLUME_RATIO_THRESHOLD)
        )
        self.closure_min_volume_voxels = float(
            gates.get("closure_min_volume_voxels", TRACKING_CLOSURE_MIN_VOLUME_VOXELS)
        )
        self.closure_strain_threshold = float(
            gates.get("closure_strain_threshold", TRACKING_CLOSURE_STRAIN_THRESHOLD)
        )

        if self.use_gpu and not HAS_GPU:
            warnings.warn("GPU acceleration requested but CuPy not available, using CPU")
            self.use_gpu = False

        self.use_hungarian = self.match_mode == "global_iou_legacy"
        self.time_series: TimeSeriesPNM = TimeSeriesPNM()
        self._reference_masks: Dict[int, Dict] = {}
        self._track_predictors: Dict[int, ConstantAccelerationKalman3D] = {}
        self._novel_segment_streaks: Dict[int, int] = {}

        algo_parts = [self.match_mode, f"solver={self.assign_solver}"]
        if self.use_gpu and self.match_mode == "legacy_greedy":
            algo_parts.append("GPU")
        if self.use_batch and self.match_mode == "legacy_greedy":
            algo_parts.append("Batch")
        print(f"[Tracker] Algorithm: {'+'.join(algo_parts)}")

    def set_reference(self, snapshot: PNMSnapshot) -> None:
        snapshot.time_index = 0
        self.time_series.reference_snapshot = snapshot
        self.time_series.snapshots = [snapshot]

        self.time_series.tracking = PoreTrackingResult(reference_ids=snapshot.pore_ids.tolist())
        self._track_predictors.clear()
        self._novel_segment_streaks.clear()

        for pore_id_raw, volume in zip(snapshot.pore_ids, snapshot.pore_volumes):
            pore_id = int(pore_id_raw)
            self.time_series.tracking.volume_history[pore_id] = [float(volume)]
            self.time_series.tracking.status_history[pore_id] = [PoreStatus.ACTIVE]
            self.time_series.tracking.iou_history[pore_id] = [1.0]
            self.time_series.tracking.match_confidence[pore_id] = [1.0]
            self.time_series.tracking.miss_count[pore_id] = 0
            self.time_series.tracking.unmatched_reason[pore_id] = ["reference"]
            idx = np.where(snapshot.pore_ids == pore_id)[0]
            if len(idx) > 0:
                center = snapshot.pore_centers[int(idx[0])]
                self.time_series.tracking.center_history[pore_id] = [center.tolist()]
                self._track_predictors[pore_id] = ConstantAccelerationKalman3D(
                    initial_position=center,
                    process_noise=self.kalman_process_noise,
                    measurement_noise=self.kalman_measurement_noise,
                )

        if snapshot.segmented_regions is not None:
            self._cache_reference_masks(snapshot)

        print(f"[Tracker] Reference set: {snapshot.num_pores} pores")

    def _cache_reference_masks(self, snapshot: PNMSnapshot) -> None:
        self._reference_masks.clear()
        regions = snapshot.segmented_regions
        if regions is None:
            return
        if snapshot.pore_ids is None or len(snapshot.pore_ids) == 0:
            return

        max_label = int(np.max(snapshot.pore_ids))
        if max_label <= 0:
            return

        # Single-pass bbox extraction avoids O(N_labels * volume) full-size masks.
        slices_by_label = ndimage.find_objects(regions, max_label=max_label)
        for pore_id_raw in snapshot.pore_ids:
            pore_id = int(pore_id_raw)
            if pore_id <= 0 or pore_id > len(slices_by_label):
                continue

            bbox_slices = slices_by_label[pore_id - 1]
            if bbox_slices is None:
                continue

            local_region = regions[bbox_slices]
            local_mask = local_region == pore_id
            if not np.any(local_mask):
                continue

            mins = np.asarray([sl.start for sl in bbox_slices], dtype=np.int32)
            maxs = np.asarray([sl.stop for sl in bbox_slices], dtype=np.int32)
            self._reference_masks[pore_id] = {
                "bbox": (mins, maxs),
                "mask": local_mask,
            }

    def _predict_center(self, ref_id: int, fallback: np.ndarray) -> np.ndarray:
        predictor = self._track_predictors.get(ref_id)
        if predictor is not None:
            return predictor.predict()

        history = self.time_series.tracking.center_history.get(ref_id, [])
        if len(history) >= 2:
            c_prev = np.asarray(history[-2], dtype=np.float64)
            c_last = np.asarray(history[-1], dtype=np.float64)
            return c_last + (c_last - c_prev)
        if len(history) == 1:
            return np.asarray(history[-1], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64)

    def _build_current_maps(self, snapshot: PNMSnapshot) -> Dict[str, object]:
        current_center_map: Dict[int, np.ndarray] = {}
        current_volume_map: Dict[int, float] = {}
        if snapshot.pore_ids is None:
            return {"center_map": current_center_map, "volume_map": current_volume_map}

        if snapshot.pore_centers is not None:
            for i, pid in enumerate(snapshot.pore_ids):
                current_center_map[int(pid)] = np.asarray(snapshot.pore_centers[i], dtype=np.float64)
        if snapshot.pore_volumes is not None:
            for i, pid in enumerate(snapshot.pore_ids):
                current_volume_map[int(pid)] = float(snapshot.pore_volumes[i])
        return {"center_map": current_center_map, "volume_map": current_volume_map}

    def _build_novel_segmentation_stats(
        self,
        snapshot: PNMSnapshot,
        id_map: Dict[int, int],
    ) -> Dict[str, Any]:
        """
        Compute diagnostics for current segments not consumed by fixed reference tracks.
        """
        current_ids = (
            [int(pid) for pid in np.asarray(snapshot.pore_ids, dtype=np.int32).tolist()]
            if snapshot.pore_ids is not None
            else []
        )
        current_volumes = (
            np.asarray(snapshot.pore_volumes, dtype=np.float64).tolist()
            if snapshot.pore_volumes is not None
            else [0.0] * len(current_ids)
        )
        volume_map: Dict[int, float] = {}
        for idx, pid in enumerate(current_ids):
            if pid > 0:
                volume_map[pid] = float(current_volumes[idx]) if idx < len(current_volumes) else 0.0

        matched_curr_ids = {int(cid) for cid in id_map.values() if int(cid) > 0}
        unmatched_curr_ids = sorted(pid for pid in volume_map.keys() if pid not in matched_curr_ids)
        above_volume = [
            pid for pid in unmatched_curr_ids if float(volume_map.get(pid, 0.0)) >= float(self.novel_min_volume_voxels)
        ]

        active_now = set(above_volume)
        for pid in list(self._novel_segment_streaks.keys()):
            if pid not in active_now:
                self._novel_segment_streaks.pop(pid, None)
        for pid in above_volume:
            self._novel_segment_streaks[pid] = int(self._novel_segment_streaks.get(pid, 0)) + 1

        persistent_ids = sorted(
            pid for pid in above_volume if int(self._novel_segment_streaks.get(pid, 0)) >= int(self.novel_min_persistence)
        )

        return {
            "time_index": int(snapshot.time_index),
            "num_current_segments": int(len(volume_map)),
            "num_matched_current_segments": int(len(matched_curr_ids)),
            "num_unmatched_current_segments": int(len(unmatched_curr_ids)),
            "num_untracked_novel_segments": int(len(persistent_ids)),
            "novel_candidate_ids": persistent_ids,
            "min_volume_voxels": float(self.novel_min_volume_voxels),
            "min_persistence": int(self.novel_min_persistence),
            "reference_policy": "fixed_reference_set",
        }

    def _estimate_macro_registration(
        self,
        previous_snapshot: Optional[PNMSnapshot],
        current_snapshot: PNMSnapshot,
    ) -> MacroRegistrationResult:
        if not self.enable_macro_registration:
            return MacroRegistrationResult(
                displacement=np.zeros(3, dtype=np.float64),
                method="disabled",
                confidence=0.0,
            )

        previous_regions = previous_snapshot.segmented_regions if previous_snapshot is not None else None
        current_regions = current_snapshot.segmented_regions
        return estimate_macro_registration(
            reference_regions=previous_regions,
            current_regions=current_regions,
            smoothing_sigma=self.macro_reg_smoothing_sigma,
            upsample_factor=self.macro_reg_upsample_factor,
            use_gpu=(self.use_gpu and self.macro_reg_use_gpu),
            gpu_min_size_mb=self.macro_reg_gpu_min_mb,
        )

    def _compose_expected_center(
        self,
        ref_id: int,
        motion_prediction: np.ndarray,
        macro_registration: MacroRegistrationResult,
    ) -> np.ndarray:
        """
        Compose gating center from KF prediction and macro displacement.

        To avoid drift, macro shift is weakly blended for well-tracked pores and
        only strongly applied after consecutive misses.
        """
        base = np.asarray(motion_prediction, dtype=np.float64)
        confidence = float(np.clip(macro_registration.confidence, 0.0, 1.0))
        misses = int(self.time_series.tracking.miss_count.get(ref_id, 0))
        history_len = len(self.time_series.tracking.center_history.get(ref_id, []))

        if confidence <= 0.0:
            return base

        if history_len <= 1:
            # Cold start: rely on macro field to survive first-frame large translations.
            macro_weight = 1.00
        elif misses <= 0:
            # Stable tracking: avoid injecting macro-field noise into KF trajectory.
            macro_weight = 0.00
        elif misses == 1:
            macro_weight = 0.50 * confidence
        else:
            macro_weight = 1.00 * confidence

        ref_snapshot = self.time_series.reference_snapshot
        spacing_xyz = getattr(ref_snapshot, "spacing", (1.0, 1.0, 1.0))
        macro_delta_world = voxel_delta_zyx_to_world_delta_xyz(
            delta_zyx=(
                float(macro_registration.displacement[0]),
                float(macro_registration.displacement[1]),
                float(macro_registration.displacement[2]),
            ),
            spacing_xyz=spacing_xyz,
        )
        return base + macro_weight * np.asarray(macro_delta_world, dtype=np.float64)

    def _apply_closure_classification(
        self,
        reference_snapshot: PNMSnapshot,
        match_results: Dict[int, Dict[str, object]],
        predicted_centers: Dict[int, np.ndarray],
        macro_registration: MacroRegistrationResult,
    ) -> None:
        tracking = self.time_series.tracking
        ref_volume_map = {
            int(pid): float(vol)
            for pid, vol in zip(reference_snapshot.pore_ids, reference_snapshot.pore_volumes)
        }
        ref_center_map = {
            int(pid): np.asarray(center, dtype=np.float64)
            for pid, center in zip(reference_snapshot.pore_ids, reference_snapshot.pore_centers)
        }

        for ref_id_raw in reference_snapshot.pore_ids:
            ref_id = int(ref_id_raw)
            info = match_results.get(ref_id)
            if info is None:
                continue
            if int(info.get("matched_id", -1)) > 0:
                continue

            ref_volume = ref_volume_map.get(ref_id, 0.0)
            prev_series = tracking.volume_history.get(ref_id, [])
            previous_volume = float(prev_series[-1]) if prev_series else ref_volume
            predicted_center = np.asarray(
                predicted_centers.get(ref_id, ref_center_map.get(ref_id, np.zeros(3, dtype=np.float64))),
                dtype=np.float64,
            )
            predicted_center_zyx = world_xyz_to_voxel_zyx(
                world_xyz=(float(predicted_center[0]), float(predicted_center[1]), float(predicted_center[2])),
                spacing_xyz=reference_snapshot.spacing,
                origin_xyz=reference_snapshot.origin,
            )
            local_compression = macro_registration.sample_compression(
                np.asarray(predicted_center_zyx, dtype=np.float64),
                default=0.0,
            )

            if should_mark_closed_by_compression(
                previous_volume=previous_volume,
                reference_volume=ref_volume,
                local_compression=local_compression,
                volume_ratio_threshold=self.closure_volume_ratio_threshold,
                min_volume_voxels=self.closure_min_volume_voxels,
                compression_threshold=self.closure_strain_threshold,
            ):
                info["reason"] = "closed_by_compression"
                info["local_compression"] = float(local_compression)
                info["matched_id"] = -1
                match_results[ref_id] = info

    def track_snapshot(
        self,
        snapshot: PNMSnapshot,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        if self.time_series.reference_snapshot is None:
            raise ValueError("Reference snapshot not set. Call set_reference first.")
        if snapshot.segmented_regions is None:
            raise ValueError("Snapshot must include segmented_regions for tracking.")
        if not self._reference_masks:
            raise ValueError("Reference masks are missing. Ensure reference snapshot includes segmented_regions.")

        start_time = time.time()
        time_index = len(self.time_series.snapshots)
        snapshot.time_index = time_index

        if self.match_mode == "legacy_greedy":
            id_map = self._track_snapshot_legacy(snapshot, callback)
            algo_desc = "legacy_greedy"
        elif self.match_mode == "global_iou_legacy":
            id_map = self._track_snapshot_global_iou(snapshot, callback)
            algo_desc = "global_iou_legacy"
        else:
            id_map = self._track_snapshot_temporal_global(snapshot, callback)
            algo_desc = "temporal_global"

        snapshot.metadata["novel_segmentation"] = self._build_novel_segmentation_stats(
            snapshot=snapshot,
            id_map=id_map,
        )
        self.time_series.tracking.id_mapping[time_index] = id_map
        self.time_series.snapshots.append(snapshot)

        num_pores = len(self.time_series.reference_snapshot.pore_ids)
        num_compressed = sum(
            1
            for statuses in self.time_series.tracking.status_history.values()
            if statuses and statuses[-1] == PoreStatus.COMPRESSED
        )
        num_active = num_pores - num_compressed

        elapsed = time.time() - start_time
        print(
            f"[Tracker] t={time_index}: {num_active} active, {num_compressed} compressed "
            f"({elapsed:.2f}s, {algo_desc})"
        )
        if callback:
            callback(100, f"Tracked {num_pores} pores")

    def _track_snapshot_temporal_global(
        self,
        snapshot: PNMSnapshot,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[int, int]:
        ref_snapshot = self.time_series.reference_snapshot
        if ref_snapshot is None:
            return {}

        previous_snapshot = self.time_series.snapshots[-1] if self.time_series.snapshots else ref_snapshot
        macro_registration = self._estimate_macro_registration(previous_snapshot, snapshot)
        if callback:
            callback(20, f"Macro registration: {macro_registration.method}")

        maps = self._build_current_maps(snapshot)
        current_center_map = maps["center_map"]
        current_volume_map = maps["volume_map"]

        predicted_centers: Dict[int, np.ndarray] = {}
        for i, ref_id_raw in enumerate(ref_snapshot.pore_ids):
            ref_id = int(ref_id_raw)
            motion_prediction = self._predict_center(ref_id, ref_snapshot.pore_centers[i])
            predicted_centers[ref_id] = self._compose_expected_center(
                ref_id=ref_id,
                motion_prediction=motion_prediction,
                macro_registration=macro_registration,
            )

        if callback:
            callback(35, "Building TGGA candidates...")

        candidate_data = build_candidates(
            reference_snapshot=ref_snapshot,
            current_snapshot=snapshot,
            reference_masks=self._reference_masks,
            current_regions=snapshot.segmented_regions,
            spacing_xyz=ref_snapshot.spacing,
            predicted_centers=predicted_centers,
            cost_weights=self.cost_weights,
            gate_center_radius_factor=self.gate_center_radius_factor,
            gate_volume_ratio_min=self.gate_volume_ratio_min,
            gate_volume_ratio_max=self.gate_volume_ratio_max,
            gate_iou_min=self.gate_iou_min,
            gate_volume_ratio_min_floor=self.gate_volume_ratio_min_floor,
            small_pore_volume_voxels=self.small_pore_volume_voxels,
            soft_gate_min_intersection_voxels=self.soft_gate_min_intersection_voxels,
            soft_gate_cost_penalty=self.soft_gate_cost_penalty,
            volume_cost_mode=self.volume_cost_mode,
            volume_cost_sigma=self.volume_cost_sigma,
        )

        cost_matrix = candidate_data["cost_matrix"]
        pair_metrics = candidate_data["pair_metrics"]
        row_geom = candidate_data["row_geom_candidate"]
        row_valid = candidate_data["row_valid_candidate"]
        curr_ids = (
            np.asarray(snapshot.pore_ids, dtype=np.int32)
            if snapshot.pore_ids is not None
            else np.array([], dtype=np.int32)
        )

        if callback:
            callback(55, "Solving global assignment...")

        assignments, solver_used = solve_global_assignment(
            cost_matrix=cost_matrix,
            assign_solver=self.assign_solver,
            invalid_cost=INVALID_COST,
        )

        match_results: Dict[int, Dict[str, object]] = {}
        for ref_idx, ref_id_raw in enumerate(ref_snapshot.pore_ids):
            ref_id = int(ref_id_raw)
            if not bool(row_geom[ref_idx]):
                reason = "gate_rejected"
            elif not bool(row_valid[ref_idx]):
                reason = "iou_below_gate"
            else:
                reason = "not_selected_by_global_assignment"
            match_results[ref_id] = {"matched_id": -1, "reason": reason}

        for ref_idx, curr_idx, cost in assignments:
            metrics = pair_metrics.get((ref_idx, curr_idx))
            if metrics is None:
                continue
            ref_id = int(ref_snapshot.pore_ids[ref_idx])
            curr_id = int(curr_ids[curr_idx]) if curr_idx < len(curr_ids) else -1
            match_results[ref_id] = {
                "matched_id": curr_id,
                "iou": float(metrics["iou"]),
                "dice_local": float(metrics["dice_local"]),
                "volume_ratio": float(metrics["volume_ratio"]),
                "current_volume": float(current_volume_map.get(curr_id, 0.0)),
                "cost": float(cost),
                "reason": f"matched_{solver_used}",
            }

        gate_rejected = 0
        iou_below_gate = 0
        not_selected = 0
        matched = 0
        matched_curr_ids: set[int] = set()
        for info in match_results.values():
            reason = str(info.get("reason", ""))
            if reason == "gate_rejected":
                gate_rejected += 1
            elif reason == "iou_below_gate":
                iou_below_gate += 1
            elif reason.startswith("matched_"):
                matched += 1
                curr_id = int(info.get("matched_id", -1))
                if curr_id > 0:
                    matched_curr_ids.add(curr_id)
            else:
                not_selected += 1
        snapshot.metadata["tgga_debug"] = {
            "time_index": int(snapshot.time_index),
            "solver": str(solver_used),
            "num_assignments": int(len(assignments)),
            "gate_rejected": int(gate_rejected),
            "iou_below_gate": int(iou_below_gate),
            "not_selected": int(not_selected),
            "matched": int(matched),
            "num_current_segments": int(len(curr_ids)),
            "num_unmatched_current_segments": int(max(0, len(curr_ids) - len(matched_curr_ids))),
        }

        self._apply_closure_classification(
            reference_snapshot=ref_snapshot,
            match_results=match_results,
            predicted_centers=predicted_centers,
            macro_registration=macro_registration,
        )

        id_map = update_tracks_with_hysteresis(
            tracking=self.time_series.tracking,
            reference_snapshot=ref_snapshot,
            current_center_map=current_center_map,
            match_results=match_results,
            max_misses=self.max_misses,
            iou_threshold=self.iou_threshold,
            compression_threshold=self.compression_threshold,
            center_smoothing=self.center_smoothing,
            predicted_centers=predicted_centers,
            track_predictors=self._track_predictors,
            kalman_brake_velocity_decay=self.kalman_brake_velocity_decay,
            kalman_brake_acceleration_decay=self.kalman_brake_acceleration_decay,
            kalman_freeze_after_misses=self.kalman_freeze_after_misses,
        )
        if callback:
            callback(85, f"TGGA matched {len(assignments)} tracks ({solver_used})")
        return id_map

    def _build_global_iou_matrix(self, snapshot: PNMSnapshot) -> Tuple[np.ndarray, Dict[int, float]]:
        ref_snapshot = self.time_series.reference_snapshot
        if ref_snapshot is None:
            return np.zeros((0, 0), dtype=np.float64), {}

        ref_ids = np.asarray(ref_snapshot.pore_ids, dtype=np.int32)
        curr_ids = (
            np.asarray(snapshot.pore_ids, dtype=np.int32)
            if snapshot.pore_ids is not None
            else np.array([], dtype=np.int32)
        )
        iou_matrix = np.zeros((len(ref_ids), len(curr_ids)), dtype=np.float64)

        curr_volume_map: Dict[int, float] = {}
        if snapshot.pore_volumes is not None and snapshot.pore_ids is not None:
            for i, pid in enumerate(snapshot.pore_ids):
                curr_volume_map[int(pid)] = float(snapshot.pore_volumes[i])

        if len(curr_ids) == 0:
            return iou_matrix, curr_volume_map

        curr_index_map = {int(pid): idx for idx, pid in enumerate(curr_ids)}
        current_regions = snapshot.segmented_regions

        for ref_idx, ref_id_raw in enumerate(ref_ids):
            ref_id = int(ref_id_raw)
            mask_data = self._reference_masks.get(ref_id)
            if mask_data is None:
                continue
            mins, maxs = mask_data["bbox"]
            local_mask = mask_data["mask"]
            current_region = current_regions[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            overlaps = current_region[local_mask]
            overlaps = overlaps[overlaps > 0]
            if len(overlaps) == 0:
                continue
            unique_labels = np.unique(overlaps)
            for label in unique_labels:
                curr_idx = curr_index_map.get(int(label))
                if curr_idx is None:
                    continue
                iou, _dice, _local_vol, _intersection = _compute_overlap_metrics(local_mask, current_region, int(label))
                iou_matrix[ref_idx, curr_idx] = max(iou_matrix[ref_idx, curr_idx], iou)

        return iou_matrix, curr_volume_map

    def _track_snapshot_global_iou(
        self,
        snapshot: PNMSnapshot,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[int, int]:
        ref_snapshot = self.time_series.reference_snapshot
        if ref_snapshot is None:
            return {}

        if callback:
            callback(25, "Building global IoU matrix...")

        previous_snapshot = self.time_series.snapshots[-1] if self.time_series.snapshots else ref_snapshot
        macro_registration = self._estimate_macro_registration(previous_snapshot, snapshot)
        predicted_centers: Dict[int, np.ndarray] = {}
        for i, ref_id_raw in enumerate(ref_snapshot.pore_ids):
            ref_id = int(ref_id_raw)
            motion_prediction = self._predict_center(ref_id, ref_snapshot.pore_centers[i])
            predicted_centers[ref_id] = self._compose_expected_center(
                ref_id=ref_id,
                motion_prediction=motion_prediction,
                macro_registration=macro_registration,
            )

        maps = self._build_current_maps(snapshot)
        current_center_map = maps["center_map"]

        iou_matrix, current_volume_map = self._build_global_iou_matrix(snapshot)
        if iou_matrix.size == 0:
            match_results = {
                int(ref_id): {"matched_id": -1, "reason": "empty_iou_matrix"}
                for ref_id in ref_snapshot.pore_ids
            }
            return update_tracks_with_hysteresis(
                tracking=self.time_series.tracking,
                reference_snapshot=ref_snapshot,
                current_center_map=current_center_map,
                match_results=match_results,
                max_misses=self.max_misses,
                iou_threshold=self.iou_threshold,
                compression_threshold=self.compression_threshold,
                center_smoothing=self.center_smoothing,
                predicted_centers=predicted_centers,
                track_predictors=self._track_predictors,
                kalman_brake_velocity_decay=self.kalman_brake_velocity_decay,
                kalman_brake_acceleration_decay=self.kalman_brake_acceleration_decay,
                kalman_freeze_after_misses=self.kalman_freeze_after_misses,
            )

        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < self.iou_threshold] = INVALID_COST
        assignments, solver_used = solve_global_assignment(
            cost_matrix=cost_matrix,
            assign_solver=self.assign_solver,
            invalid_cost=INVALID_COST,
        )

        curr_ids = (
            np.asarray(snapshot.pore_ids, dtype=np.int32)
            if snapshot.pore_ids is not None
            else np.array([], dtype=np.int32)
        )
        ref_vol_map = {int(pid): float(v) for pid, v in zip(ref_snapshot.pore_ids, ref_snapshot.pore_volumes)}

        match_results: Dict[int, Dict[str, object]] = {
            int(ref_id): {"matched_id": -1, "reason": "iou_below_threshold"}
            for ref_id in ref_snapshot.pore_ids
        }
        for ref_idx, curr_idx, cost in assignments:
            if ref_idx >= iou_matrix.shape[0] or curr_idx >= iou_matrix.shape[1]:
                continue
            iou = float(iou_matrix[ref_idx, curr_idx])
            if iou < self.iou_threshold:
                continue
            ref_id = int(ref_snapshot.pore_ids[ref_idx])
            curr_id = int(curr_ids[curr_idx])
            ref_volume = ref_vol_map.get(ref_id, 0.0)
            curr_volume = float(current_volume_map.get(curr_id, 0.0))
            volume_ratio = curr_volume / ref_volume if ref_volume > 0 else 0.0
            match_results[ref_id] = {
                "matched_id": curr_id,
                "iou": iou,
                "dice_local": iou,
                "volume_ratio": volume_ratio,
                "current_volume": curr_volume,
                "cost": float(cost),
                "reason": f"matched_{solver_used}",
            }

        self._apply_closure_classification(
            reference_snapshot=ref_snapshot,
            match_results=match_results,
            predicted_centers=predicted_centers,
            macro_registration=macro_registration,
        )

        if callback:
            callback(80, f"Global IoU matched {len(assignments)} tracks ({solver_used})")
        return update_tracks_with_hysteresis(
            tracking=self.time_series.tracking,
            reference_snapshot=ref_snapshot,
            current_center_map=current_center_map,
            match_results=match_results,
            max_misses=self.max_misses,
            iou_threshold=self.iou_threshold,
            compression_threshold=self.compression_threshold,
            center_smoothing=self.center_smoothing,
            predicted_centers=predicted_centers,
            track_predictors=self._track_predictors,
            kalman_brake_velocity_decay=self.kalman_brake_velocity_decay,
            kalman_brake_acceleration_decay=self.kalman_brake_acceleration_decay,
            kalman_freeze_after_misses=self.kalman_freeze_after_misses,
        )

    def _track_snapshot_legacy(
        self,
        snapshot: PNMSnapshot,
        callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[int, int]:
        ref_snapshot = self.time_series.reference_snapshot
        if ref_snapshot is None:
            return {}
        current_regions = snapshot.segmented_regions

        current_center_map: Dict[int, np.ndarray] = {}
        if snapshot.pore_centers is not None and snapshot.pore_ids is not None:
            for i, pid in enumerate(snapshot.pore_ids):
                current_center_map[int(pid)] = snapshot.pore_centers[i]

        id_map: Dict[int, int] = {}
        num_pores = len(ref_snapshot.pore_ids)
        use_gpu_now = self.use_gpu and num_pores >= TRACKING_GPU_MIN_PORES and HAS_GPU
        use_batch_now = self.use_batch and num_pores >= 50

        if use_batch_now:
            if callback:
                callback(30, f"Computing IoU batch (GPU={use_gpu_now})...")
            ref_masks_list = [self._reference_masks[int(pid)] for pid in ref_snapshot.pore_ids]
            if use_gpu_now:
                matched_ids, iou_scores, volumes = compute_batch_iou_gpu(
                    ref_masks_list, current_regions, ref_snapshot.pore_ids
                )
            else:
                matched_ids, iou_scores, volumes = compute_batch_iou_cpu(
                    ref_masks_list, current_regions, ref_snapshot.pore_ids
                )
            if callback:
                callback(70, f"Processing {num_pores} pore matches...")

            for i, ref_id_raw in enumerate(ref_snapshot.pore_ids):
                ref_id = int(ref_id_raw)
                matched_id = int(matched_ids[i])
                iou = float(iou_scores[i])
                current_volume = float(volumes[i])
                ref_volume = float(ref_snapshot.pore_volumes[i])
                volume_ratio = current_volume / ref_volume if ref_volume > 0 else 0.0

                if iou < self.iou_threshold or volume_ratio < self.compression_threshold:
                    status = PoreStatus.COMPRESSED
                else:
                    status = PoreStatus.ACTIVE

                tracking = self.time_series.tracking
                tracking.volume_history.setdefault(ref_id, []).append(current_volume)
                tracking.status_history.setdefault(ref_id, []).append(status)
                tracking.iou_history.setdefault(ref_id, []).append(iou)
                tracking.match_confidence.setdefault(ref_id, []).append(float(np.clip(iou, 0.0, 1.0)))
                tracking.unmatched_reason.setdefault(ref_id, []).append("matched" if matched_id > 0 else "no_overlap")
                if matched_id > 0:
                    tracking.miss_count[ref_id] = 0
                else:
                    tracking.miss_count[ref_id] = int(tracking.miss_count.get(ref_id, 0)) + 1
                id_map[ref_id] = matched_id if matched_id > 0 else -1

                history = tracking.center_history.setdefault(ref_id, [])
                prev_center = np.array(history[-1]) if history else ref_snapshot.pore_centers[i]
                if matched_id > 0 and matched_id in current_center_map:
                    curr_center = current_center_map[matched_id]
                    smoothed = prev_center + self.center_smoothing * (curr_center - prev_center)
                    history.append(smoothed.tolist())
                else:
                    history.append(prev_center.tolist())
        else:
            total_pores = len(ref_snapshot.pore_ids)
            for i, ref_id_raw in enumerate(ref_snapshot.pore_ids):
                if callback and i % 50 == 0:
                    progress = int(30 + 60 * i / max(total_pores, 1))
                    callback(progress, f"Tracking pore {i+1}/{total_pores}...")

                ref_id = int(ref_id_raw)
                ref_volume = float(ref_snapshot.pore_volumes[i])
                mask_data = self._reference_masks[ref_id]
                mins, maxs = mask_data["bbox"]
                local_mask = mask_data["mask"]
                current_region = current_regions[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
                matched_id, iou, current_volume = self._match_pore(local_mask, current_region)
                volume_ratio = current_volume / ref_volume if ref_volume > 0 else 0.0

                if matched_id is None or iou < self.iou_threshold:
                    status = PoreStatus.COMPRESSED
                elif volume_ratio < self.compression_threshold:
                    status = PoreStatus.COMPRESSED
                else:
                    status = PoreStatus.ACTIVE

                tracking = self.time_series.tracking
                tracking.volume_history.setdefault(ref_id, []).append(float(current_volume))
                tracking.status_history.setdefault(ref_id, []).append(status)
                tracking.iou_history.setdefault(ref_id, []).append(float(iou))
                tracking.match_confidence.setdefault(ref_id, []).append(float(np.clip(iou, 0.0, 1.0)))
                tracking.unmatched_reason.setdefault(ref_id, []).append("matched" if matched_id else "no_overlap")
                if matched_id:
                    tracking.miss_count[ref_id] = 0
                else:
                    tracking.miss_count[ref_id] = int(tracking.miss_count.get(ref_id, 0)) + 1
                id_map[ref_id] = int(matched_id) if matched_id else -1

                history = tracking.center_history.setdefault(ref_id, [])
                prev_center = np.array(history[-1]) if history else ref_snapshot.pore_centers[i]
                if matched_id and matched_id in current_center_map:
                    curr_center = current_center_map[matched_id]
                    smoothed = prev_center + self.center_smoothing * (curr_center - prev_center)
                    history.append(smoothed.tolist())
                else:
                    history.append(prev_center.tolist())

        return id_map

    def _match_pore(
        self,
        ref_mask: np.ndarray,
        current_labels: np.ndarray,
    ) -> Tuple[Optional[int], float, float]:
        overlapping_labels = current_labels[ref_mask]
        overlapping_labels = overlapping_labels[overlapping_labels > 0]
        if len(overlapping_labels) == 0:
            return None, 0.0, 0.0

        label_counts = np.bincount(overlapping_labels)
        if len(label_counts) > 1:
            best_label = int(np.argmax(label_counts[1:]) + 1)
        else:
            best_label = int(overlapping_labels[0])

        current_mask = current_labels == best_label
        intersection = np.sum(ref_mask & current_mask)
        union = np.sum(ref_mask | current_mask)
        if union == 0:
            return None, 0.0, 0.0
        iou = float(intersection / union)
        current_volume = float(np.sum(current_mask))
        return best_label, iou, current_volume

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator <= 0:
            return float(default)
        return float(numerator / denominator)

    @staticmethod
    def _resolve_labels_path(volume: Any) -> Optional[str]:
        metadata = getattr(volume, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        sim = metadata.get("sim_annotations")
        if not isinstance(sim, dict):
            return None
        files = sim.get("files")
        if not isinstance(files, dict):
            return None
        path = files.get("labels_npy")
        if isinstance(path, str) and path:
            return path
        return None

    @staticmethod
    def _resolve_annotations_payload(
        volume: Any,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        metadata = getattr(volume, "metadata", None)
        if not isinstance(metadata, dict):
            return None, None, None
        sim = metadata.get("sim_annotations")
        if not isinstance(sim, dict):
            return None, None, None

        files = sim.get("files")
        ann_path = None
        if isinstance(files, dict):
            raw_path = files.get("annotations_json")
            if isinstance(raw_path, str) and raw_path:
                ann_path = raw_path

        in_memory = sim.get("annotations")
        if isinstance(in_memory, dict):
            return in_memory, ann_path, None

        if not ann_path:
            return None, None, None
        if not os.path.isfile(ann_path):
            return None, ann_path, None

        try:
            with open(ann_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                return loaded, ann_path, None
            return None, ann_path, "annotations.json root is not an object"
        except Exception as exc:
            return None, ann_path, f"failed to parse annotations.json: {exc}"

    @staticmethod
    def _extract_annotation_voids(
        annotations: Dict[str, Any],
        default_spacing_xyz: Tuple[float, float, float],
        default_origin_xyz: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Parse per-step annotation void entries into dense numeric arrays.

        Returns:
            (ids, centers_world_xyz, radii_mm, volumes_mm3, parse_stats)
        """
        if not isinstance(annotations, dict):
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                {"num_voids": 0, "num_invalid_voids": 0},
            )

        voxel_size = annotations.get("voxel_size")
        if isinstance(voxel_size, (int, float)) and float(voxel_size) > 0:
            spacing_xyz = (float(voxel_size), float(voxel_size), float(voxel_size))
        else:
            spacing_xyz = (
                float(default_spacing_xyz[0]),
                float(default_spacing_xyz[1]),
                float(default_spacing_xyz[2]),
            )

        origin = annotations.get("origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 3:
            origin_xyz = (float(origin[0]), float(origin[1]), float(origin[2]))
        else:
            origin_xyz = (
                float(default_origin_xyz[0]),
                float(default_origin_xyz[1]),
                float(default_origin_xyz[2]),
            )

        voids = annotations.get("voids", [])
        if not isinstance(voids, list):
            voids = []

        ids: List[int] = []
        centers: List[Tuple[float, float, float]] = []
        radii: List[float] = []
        volumes: List[float] = []

        invalid_count = 0
        seen_ids: set[int] = set()
        sx, sy, sz = spacing_xyz
        ox, oy, oz = origin_xyz

        for row in voids:
            if not isinstance(row, dict):
                invalid_count += 1
                continue

            raw_id = row.get("id")
            if not isinstance(raw_id, (int, np.integer)):
                invalid_count += 1
                continue
            gt_id = int(raw_id)
            if gt_id <= 0 or gt_id in seen_ids:
                invalid_count += 1
                continue

            center_world = row.get("center_mm")
            if isinstance(center_world, (list, tuple)) and len(center_world) == 3:
                cx = float(center_world[0])
                cy = float(center_world[1])
                cz = float(center_world[2])
            else:
                center_voxel = row.get("center_voxel")
                if not (isinstance(center_voxel, (list, tuple)) and len(center_voxel) == 3):
                    invalid_count += 1
                    continue
                vx = float(center_voxel[0])
                vy = float(center_voxel[1])
                vz = float(center_voxel[2])
                cx = ox + vx * sx
                cy = oy + vy * sy
                cz = oz + vz * sz

            radius_raw = row.get("radius_mm")
            if isinstance(radius_raw, (int, float)) and float(radius_raw) > 0:
                radius_mm = float(radius_raw)
            else:
                volume_raw = row.get("volume_mm3")
                if isinstance(volume_raw, (int, float)) and float(volume_raw) > 0:
                    radius_mm = float((3.0 * float(volume_raw) / (4.0 * np.pi)) ** (1.0 / 3.0))
                else:
                    radius_mm = 0.0

            volume_raw = row.get("volume_mm3")
            if isinstance(volume_raw, (int, float)) and float(volume_raw) >= 0:
                volume_mm3 = float(volume_raw)
            else:
                volume_mm3 = float((4.0 / 3.0) * np.pi * max(radius_mm, 0.0) ** 3)

            seen_ids.add(gt_id)
            ids.append(gt_id)
            centers.append((cx, cy, cz))
            radii.append(float(radius_mm))
            volumes.append(float(volume_mm3))

        return (
            np.asarray(ids, dtype=np.int32),
            np.asarray(centers, dtype=np.float64).reshape((-1, 3)) if centers else np.zeros((0, 3), dtype=np.float64),
            np.asarray(radii, dtype=np.float64),
            np.asarray(volumes, dtype=np.float64),
            {"num_voids": int(len(ids)), "num_invalid_voids": int(invalid_count)},
        )

    def _evaluate_step_against_annotations(
        self,
        snapshot: PNMSnapshot,
        annotations: Dict[str, Any],
        instance_iou_threshold: float,
    ) -> Dict[str, Any]:
        """
        Evaluate one snapshot against annotation-defined pore entities.

        Matching is solved as one-to-one assignment with center/radius/volume
        costs under distance gates, producing pore-level detection metrics.
        """
        pred_ids = (
            np.asarray(snapshot.pore_ids, dtype=np.int32)
            if snapshot.pore_ids is not None
            else np.zeros((0,), dtype=np.int32)
        )
        pred_centers = (
            np.asarray(snapshot.pore_centers, dtype=np.float64)
            if snapshot.pore_centers is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        pred_radii = (
            np.asarray(snapshot.pore_radii, dtype=np.float64)
            if snapshot.pore_radii is not None
            else np.zeros((len(pred_ids),), dtype=np.float64)
        )
        pred_volumes = (
            np.asarray(snapshot.pore_volumes, dtype=np.float64)
            if snapshot.pore_volumes is not None
            else np.zeros((len(pred_ids),), dtype=np.float64)
        )

        if pred_centers.ndim != 2 or pred_centers.shape[1] != 3 or pred_centers.shape[0] != len(pred_ids):
            pred_centers = np.zeros((len(pred_ids), 3), dtype=np.float64)
        if pred_radii.shape[0] != len(pred_ids):
            pred_radii = np.zeros((len(pred_ids),), dtype=np.float64)
        if pred_volumes.shape[0] != len(pred_ids):
            pred_volumes = np.zeros((len(pred_ids),), dtype=np.float64)

        spacing_xyz = (
            float(snapshot.spacing[0]) if snapshot.spacing else 1.0,
            float(snapshot.spacing[1]) if snapshot.spacing else 1.0,
            float(snapshot.spacing[2]) if snapshot.spacing else 1.0,
        )
        origin_xyz = (
            float(snapshot.origin[0]) if snapshot.origin else 0.0,
            float(snapshot.origin[1]) if snapshot.origin else 0.0,
            float(snapshot.origin[2]) if snapshot.origin else 0.0,
        )
        voxel_volume_mm3 = max(abs(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]), 1e-12)
        pred_volumes_mm3 = pred_volumes * voxel_volume_mm3

        gt_ids, gt_centers, gt_radii, gt_volumes, parse_stats = self._extract_annotation_voids(
            annotations=annotations,
            default_spacing_xyz=spacing_xyz,
            default_origin_xyz=origin_xyz,
        )

        n_pred = int(len(pred_ids))
        n_gt = int(len(gt_ids))
        if n_pred == 0 and n_gt == 0:
            return {
                "segmentation": {},
                "instance": {
                    "num_pred_instances": 0,
                    "num_gt_instances": 0,
                    "num_matches": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "mean_matched_iou": 0.0,
                    "iou_threshold": float(instance_iou_threshold),
                    "assignment_solver": "none",
                    "matching_mode": "annotations_center",
                    "mean_center_distance_mm": 0.0,
                },
                "mapping": {"pred_to_gt": {}, "gt_to_pred": {}, "gt_ids": set()},
                "annotation": parse_stats,
            }

        cost_matrix = np.full((n_pred, n_gt), INVALID_COST, dtype=np.float64)
        center_dist_matrix = np.full((n_pred, n_gt), np.inf, dtype=np.float64)
        center_score_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

        avg_spacing = max(float(np.mean(np.asarray(spacing_xyz, dtype=np.float64))), 1e-6)
        min_gate_mm = 2.0 * avg_spacing

        for i in range(n_pred):
            pred_center = pred_centers[i]
            pred_radius = float(max(pred_radii[i], 0.0))
            pred_volume_mm3 = float(max(pred_volumes_mm3[i], 0.0))
            for j in range(n_gt):
                gt_center = gt_centers[j]
                gt_radius = float(max(gt_radii[j], 0.0))
                gt_volume_mm3 = float(max(gt_volumes[j], 0.0))

                dist_mm = float(np.linalg.norm(pred_center - gt_center))
                gate_mm = float(
                    max(
                        min_gate_mm,
                        pred_radius + gt_radius,
                        2.5 * max(pred_radius, gt_radius),
                    )
                )
                if dist_mm > gate_mm:
                    continue

                center_term = float(np.clip(dist_mm / max(gate_mm, 1e-6), 0.0, 1.0))

                radius_scale = max(max(pred_radius, gt_radius), avg_spacing, 1e-6)
                radius_term = float(np.clip(abs(pred_radius - gt_radius) / radius_scale, 0.0, 1.0))

                if pred_volume_mm3 > 0.0 and gt_volume_mm3 > 0.0:
                    vol_term = float(
                        np.clip(abs(np.log(pred_volume_mm3 / gt_volume_mm3)) / np.log(4.0), 0.0, 1.0)
                    )
                elif pred_volume_mm3 == 0.0 and gt_volume_mm3 == 0.0:
                    vol_term = 0.0
                else:
                    vol_term = 1.0

                cost = 0.75 * center_term + 0.20 * radius_term + 0.05 * vol_term
                cost_matrix[i, j] = float(cost)
                center_dist_matrix[i, j] = dist_mm
                center_score_matrix[i, j] = float(max(0.0, 1.0 - center_term))

        try:
            assignments, solver_used = solve_global_assignment(
                cost_matrix=cost_matrix,
                assign_solver=self.assign_solver,
                invalid_cost=INVALID_COST,
            )
        except Exception:
            solver_used = "greedy_fallback"
            assignments = []
            valid_pairs = np.argwhere(cost_matrix < INVALID_COST * 0.5)
            scored_pairs = sorted(
                ((float(cost_matrix[i, j]), int(i), int(j)) for i, j in valid_pairs),
                key=lambda row: row[0],
            )
            used_rows = set()
            used_cols = set()
            for _cost, i, j in scored_pairs:
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                assignments.append((int(i), int(j), float(cost_matrix[i, j])))

        thr = float(max(instance_iou_threshold, 0.0))
        pred_to_gt: Dict[int, int] = {}
        gt_to_pred: Dict[int, int] = {}
        matched_scores: List[float] = []
        matched_dists: List[float] = []
        for pred_idx, gt_idx, _cost in assignments:
            if pred_idx >= n_pred or gt_idx >= n_gt:
                continue
            if cost_matrix[pred_idx, gt_idx] >= INVALID_COST * 0.5:
                continue
            center_score = float(center_score_matrix[pred_idx, gt_idx])
            if center_score < thr:
                continue
            pred_id = int(pred_ids[pred_idx])
            gt_id = int(gt_ids[gt_idx])
            pred_to_gt[pred_id] = gt_id
            gt_to_pred[gt_id] = pred_id
            matched_scores.append(center_score)
            matched_dists.append(float(center_dist_matrix[pred_idx, gt_idx]))

        num_matches = int(len(pred_to_gt))
        if n_pred == 0 and n_gt == 0:
            inst_precision = 1.0
            inst_recall = 1.0
            inst_f1 = 1.0
        else:
            inst_precision = self._safe_ratio(num_matches, n_pred, default=0.0)
            inst_recall = self._safe_ratio(num_matches, n_gt, default=0.0)
            inst_f1 = self._safe_ratio(
                2.0 * inst_precision * inst_recall,
                inst_precision + inst_recall,
                default=0.0,
            )

        return {
            "segmentation": {},
            "instance": {
                "num_pred_instances": int(n_pred),
                "num_gt_instances": int(n_gt),
                "num_matches": int(num_matches),
                "precision": float(inst_precision),
                "recall": float(inst_recall),
                "f1": float(inst_f1),
                # Keep legacy key name for compatibility with existing UI/tests.
                "mean_matched_iou": float(np.mean(matched_scores)) if matched_scores else 0.0,
                "iou_threshold": float(thr),
                "assignment_solver": solver_used,
                "matching_mode": "annotations_center",
                "mean_center_distance_mm": float(np.mean(matched_dists)) if matched_dists else 0.0,
            },
            "mapping": {
                "pred_to_gt": pred_to_gt,
                "gt_to_pred": gt_to_pred,
                "gt_ids": set(int(v) for v in gt_ids.tolist()),
            },
            "annotation": parse_stats,
        }

    @staticmethod
    def _build_gt_to_pore_group_map(
        predicted_labels_t0: np.ndarray,
        gt_labels_t0: np.ndarray,
    ) -> Dict[int, int]:
        """
        Build many-to-one GT-id -> pore-group mapping using t0 overlap.

        Each GT instance is assigned to the predicted pore id with maximum voxel
        overlap at t0. This collapses fine-grained GT instances into pore-level
        groups while keeping deterministic tie-breaking.
        """
        pred = np.asarray(predicted_labels_t0)
        gt = np.asarray(gt_labels_t0)
        overlap_mask = (pred > 0) & (gt > 0)
        if not np.any(overlap_mask):
            return {}

        pred_overlap = pred[overlap_mask].astype(np.uint64, copy=False)
        gt_overlap = gt[overlap_mask].astype(np.uint64, copy=False)
        pair_keys = (gt_overlap << np.uint64(32)) | pred_overlap
        pair_unique, pair_counts = np.unique(pair_keys, return_counts=True)

        best_by_gt: Dict[int, Tuple[int, int]] = {}
        for pair_key, count_raw in zip(pair_unique, pair_counts):
            key_int = int(pair_key)
            gt_id = int(key_int >> 32)
            pred_id = int(key_int & 0xFFFFFFFF)
            count = int(count_raw)
            prev = best_by_gt.get(gt_id)
            if prev is None or count > prev[1] or (count == prev[1] and pred_id < prev[0]):
                best_by_gt[gt_id] = (pred_id, count)

        return {int(gt_id): int(pred_id) for gt_id, (pred_id, _cnt) in best_by_gt.items()}

    @staticmethod
    def _remap_gt_labels_with_group_map(
        gt_labels: np.ndarray,
        gt_to_group: Dict[int, int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Remap fine-grained GT labels into pore-level group labels.

        Known GT ids use t0-derived group ids; unknown ids get deterministic
        new ids after the known-group range so they remain diagnosable as novel.
        """
        gt = np.asarray(gt_labels)
        if gt.size == 0:
            return np.zeros_like(gt, dtype=np.int32), {
                "known_gt_ids": 0,
                "unknown_gt_ids": 0,
                "num_group_ids": 0,
            }

        positive_mask = gt > 0
        if not np.any(positive_mask):
            return np.zeros_like(gt, dtype=np.int32), {
                "known_gt_ids": 0,
                "unknown_gt_ids": 0,
                "num_group_ids": 0,
            }

        unique_gt_ids = np.unique(gt[positive_mask]).astype(np.int64, copy=False)
        mapped_gt_ids = np.empty_like(unique_gt_ids, dtype=np.int32)

        known_groups = set(int(v) for v in gt_to_group.values())
        next_unknown_group = (max(known_groups) + 1) if known_groups else 1
        known_count = 0
        unknown_count = 0

        for idx, gt_id_raw in enumerate(unique_gt_ids):
            gt_id = int(gt_id_raw)
            group_id = gt_to_group.get(gt_id)
            if group_id is None:
                group_id = int(next_unknown_group)
                next_unknown_group += 1
                unknown_count += 1
            else:
                known_count += 1
            mapped_gt_ids[idx] = int(group_id)

        gt_int = gt.astype(np.int64, copy=False)
        flat_gt = gt_int.reshape(-1)
        remapped_flat = np.zeros(flat_gt.shape, dtype=np.int32)

        positive_idx = np.nonzero(flat_gt > 0)[0]
        positive_vals = flat_gt[positive_idx]
        sorted_pos = np.searchsorted(unique_gt_ids, positive_vals)
        remapped_flat[positive_idx] = mapped_gt_ids[sorted_pos]

        remapped = remapped_flat.reshape(gt.shape)
        return remapped, {
            "known_gt_ids": int(known_count),
            "unknown_gt_ids": int(unknown_count),
            "num_group_ids": int(len(set(mapped_gt_ids.tolist()))),
        }

    def _match_instance_ids(
        self,
        iou_matrix: np.ndarray,
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int, float]], str]:
        if iou_matrix.size == 0:
            return [], "none"

        thr = float(max(iou_threshold, 0.0))
        cost = 1.0 - iou_matrix
        cost[iou_matrix < thr] = INVALID_COST

        try:
            assignments, solver_used = solve_global_assignment(
                cost_matrix=cost,
                assign_solver=self.assign_solver,
                invalid_cost=INVALID_COST,
            )
            matched: List[Tuple[int, int, float]] = []
            for row_idx, col_idx, _cost in assignments:
                if row_idx >= iou_matrix.shape[0] or col_idx >= iou_matrix.shape[1]:
                    continue
                score = float(iou_matrix[row_idx, col_idx])
                if score >= thr:
                    matched.append((int(row_idx), int(col_idx), score))
            return matched, solver_used
        except Exception:
            pass

        pairs = np.argwhere(iou_matrix >= thr)
        scored_pairs: List[Tuple[float, int, int]] = [
            (float(iou_matrix[row_idx, col_idx]), int(row_idx), int(col_idx))
            for row_idx, col_idx in pairs
        ]
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        used_rows = set()
        used_cols = set()
        greedy_matches: List[Tuple[int, int, float]] = []
        for score, row_idx, col_idx in scored_pairs:
            if row_idx in used_rows or col_idx in used_cols:
                continue
            used_rows.add(row_idx)
            used_cols.add(col_idx)
            greedy_matches.append((row_idx, col_idx, score))
        return greedy_matches, "greedy_fallback"

    def _evaluate_step_against_gt(
        self,
        predicted_labels: np.ndarray,
        gt_labels: np.ndarray,
        instance_iou_threshold: float,
    ) -> Dict[str, Any]:
        pred = np.asarray(predicted_labels)
        gt = np.asarray(gt_labels)

        pred_binary = pred > 0
        gt_binary = gt > 0
        tp = int(np.count_nonzero(pred_binary & gt_binary))
        fp = int(np.count_nonzero(pred_binary & (~gt_binary)))
        fn = int(np.count_nonzero((~pred_binary) & gt_binary))

        voxel_precision = self._safe_ratio(tp, tp + fp, default=1.0 if (tp + fp + fn) == 0 else 0.0)
        voxel_recall = self._safe_ratio(tp, tp + fn, default=1.0 if (tp + fp + fn) == 0 else 0.0)
        voxel_iou = self._safe_ratio(tp, tp + fp + fn, default=1.0 if (tp + fp + fn) == 0 else 0.0)
        voxel_dice = self._safe_ratio(2.0 * tp, 2.0 * tp + fp + fn, default=1.0 if (tp + fp + fn) == 0 else 0.0)

        pred_ids_arr, pred_counts_arr = np.unique(pred[pred > 0], return_counts=True)
        gt_ids_arr, gt_counts_arr = np.unique(gt[gt > 0], return_counts=True)
        pred_ids = [int(v) for v in pred_ids_arr.tolist()]
        gt_ids = [int(v) for v in gt_ids_arr.tolist()]

        pred_count_map = {int(pid): int(cnt) for pid, cnt in zip(pred_ids_arr, pred_counts_arr)}
        gt_count_map = {int(gid): int(cnt) for gid, cnt in zip(gt_ids_arr, gt_counts_arr)}
        pred_index = {pid: idx for idx, pid in enumerate(pred_ids)}
        gt_index = {gid: idx for idx, gid in enumerate(gt_ids)}
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)

        positive_overlap = pred_binary & gt_binary
        if np.any(positive_overlap) and pred_ids and gt_ids:
            pred_overlap = pred[positive_overlap].astype(np.uint64, copy=False)
            gt_overlap = gt[positive_overlap].astype(np.uint64, copy=False)
            pair_keys = (pred_overlap << np.uint64(32)) | gt_overlap
            pair_unique, pair_counts = np.unique(pair_keys, return_counts=True)

            for pair_key, inter_count_raw in zip(pair_unique, pair_counts):
                key_int = int(pair_key)
                pred_id = int(key_int >> 32)
                gt_id = int(key_int & 0xFFFFFFFF)
                row_idx = pred_index.get(pred_id)
                col_idx = gt_index.get(gt_id)
                if row_idx is None or col_idx is None:
                    continue
                intersection = int(inter_count_raw)
                union = pred_count_map[pred_id] + gt_count_map[gt_id] - intersection
                if union > 0:
                    iou_matrix[row_idx, col_idx] = float(intersection / union)

        matches, solver_used = self._match_instance_ids(
            iou_matrix=iou_matrix,
            iou_threshold=instance_iou_threshold,
        )
        matched_ious = [float(iou) for _, _, iou in matches]

        pred_to_gt: Dict[int, int] = {}
        gt_to_pred: Dict[int, int] = {}
        for row_idx, col_idx, _iou in matches:
            if row_idx < len(pred_ids) and col_idx < len(gt_ids):
                pred_to_gt[int(pred_ids[row_idx])] = int(gt_ids[col_idx])
                gt_to_pred[int(gt_ids[col_idx])] = int(pred_ids[row_idx])

        num_pred = len(pred_ids)
        num_gt = len(gt_ids)
        num_matches = len(matches)
        if num_pred == 0 and num_gt == 0:
            inst_precision = 1.0
            inst_recall = 1.0
            inst_f1 = 1.0
        else:
            inst_precision = self._safe_ratio(num_matches, num_pred, default=0.0)
            inst_recall = self._safe_ratio(num_matches, num_gt, default=0.0)
            inst_f1 = self._safe_ratio(2.0 * inst_precision * inst_recall, inst_precision + inst_recall, default=0.0)

        return {
            "segmentation": {
                "tp_voxels": tp,
                "fp_voxels": fp,
                "fn_voxels": fn,
                "voxel_precision": voxel_precision,
                "voxel_recall": voxel_recall,
                "voxel_iou": voxel_iou,
                "voxel_dice": voxel_dice,
            },
            "instance": {
                "num_pred_instances": num_pred,
                "num_gt_instances": num_gt,
                "num_matches": num_matches,
                "precision": inst_precision,
                "recall": inst_recall,
                "f1": inst_f1,
                "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
                "iou_threshold": float(instance_iou_threshold),
                "assignment_solver": solver_used,
            },
            "mapping": {
                "pred_to_gt": pred_to_gt,
                "gt_to_pred": gt_to_pred,
                "gt_ids": set(gt_ids),
            },
        }

    def _evaluate_tracking_step(
        self,
        time_index: int,
        id_map: Dict[int, int],
        reference_pred_to_gt: Dict[int, int],
        current_pred_to_gt: Dict[int, int],
        current_gt_ids: set[int],
        novel_segmentation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evaluable_refs = len(reference_pred_to_gt)
        if evaluable_refs == 0:
            return {
                "available": False,
                "time_index": int(time_index),
                "reason": "no_reference_pred_to_gt_mapping",
                "per_reference": {},
            }

        expected_present = 0
        expected_absent = 0
        correct_active = 0
        correct_absent = 0
        missed = 0
        id_switched = 0
        unmatched_to_gt = 0
        false_positive_alive = 0
        per_reference: Dict[int, Dict[str, Any]] = {}

        for ref_pred_id, ref_gt_id in reference_pred_to_gt.items():
            matched_pred_id = int(id_map.get(int(ref_pred_id), -1))
            present = int(ref_gt_id) in current_gt_ids

            outcome = "unknown"
            mapped_gt = None
            is_correct = False
            if present:
                expected_present += 1
                if matched_pred_id <= 0:
                    missed += 1
                    outcome = "missed"
                else:
                    mapped_gt = current_pred_to_gt.get(matched_pred_id)
                    if mapped_gt is None:
                        unmatched_to_gt += 1
                        outcome = "unmatched_to_gt"
                    elif int(mapped_gt) == int(ref_gt_id):
                        correct_active += 1
                        outcome = "correct_active"
                        is_correct = True
                    else:
                        id_switched += 1
                        outcome = "id_switched"
            else:
                expected_absent += 1
                if matched_pred_id <= 0:
                    correct_absent += 1
                    outcome = "correct_absent"
                    is_correct = True
                else:
                    false_positive_alive += 1
                    mapped_gt = current_pred_to_gt.get(matched_pred_id)
                    outcome = "false_positive_alive"

            per_reference[int(ref_pred_id)] = {
                "ref_pred_id": int(ref_pred_id),
                "ref_gt_id": int(ref_gt_id),
                "present_in_gt": bool(present),
                "matched_pred_id": int(matched_pred_id),
                "mapped_gt_id": int(mapped_gt) if mapped_gt is not None else None,
                "outcome": outcome,
                "correct": bool(is_correct),
            }

        correct_total = correct_active + correct_absent
        total_gt_ids = int(len(current_gt_ids))
        reference_scope_gt_coverage = self._safe_ratio(expected_present, total_gt_ids, default=1.0 if total_gt_ids == 0 else 0.0)
        novel_stats = novel_segmentation if isinstance(novel_segmentation, dict) else {}
        untracked_novel_segments = int(novel_stats.get("num_untracked_novel_segments", 0))
        return {
            "available": True,
            "time_index": int(time_index),
            "evaluable_reference_instances": int(evaluable_refs),
            "reference_scope_total_gt_ids": int(total_gt_ids),
            "reference_scope_expected_present": int(expected_present),
            "reference_scope_gt_coverage": float(reference_scope_gt_coverage),
            "expected_present": int(expected_present),
            "expected_absent": int(expected_absent),
            "correct_active": int(correct_active),
            "correct_absent": int(correct_absent),
            "missed": int(missed),
            "id_switched": int(id_switched),
            "unmatched_to_gt": int(unmatched_to_gt),
            "untracked_novel_segments": int(untracked_novel_segments),
            "false_positive_alive": int(false_positive_alive),
            "accuracy": self._safe_ratio(correct_total, evaluable_refs, default=0.0),
            "active_accuracy": self._safe_ratio(correct_active, expected_present, default=1.0 if expected_present == 0 else 0.0),
            "closure_accuracy": self._safe_ratio(correct_absent, expected_absent, default=1.0 if expected_absent == 0 else 0.0),
            "novel_segmentation": novel_stats,
            "per_reference": per_reference,
        }

    def evaluate_against_sim_annotations(
        self,
        volumes: List[Any],
        instance_iou_threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate segmentation + tracking quality against simulation annotations.

        Primary evaluation uses `annotations.json` pore entities (`voids`) for
        pore-level one-to-one detection/tracking. When `annotations.json` is
        unavailable, the method falls back to legacy `labels.npy`-based logic.
        """
        report: Dict[str, Any] = {
            "available": False,
            "status": "unavailable",
            "instance_iou_threshold": float(max(instance_iou_threshold, 0.0)),
            "steps": [],
            "overall": {},
            "errors": [],
            "warnings": [],
        }

        snapshots = self.time_series.snapshots
        if not snapshots:
            report["errors"].append("No snapshots available for evaluation")
            self.time_series.tracking.evaluation = report
            return report

        if not volumes:
            report["errors"].append("No loaded volumes available for evaluation")
            self.time_series.tracking.evaluation = report
            return report

        steps_to_eval = min(len(snapshots), len(volumes))
        if steps_to_eval == 0:
            report["errors"].append("No overlapping steps between snapshots and volumes")
            self.time_series.tracking.evaluation = report
            return report

        if len(snapshots) != len(volumes):
            report["warnings"].append(
                f"snapshot count ({len(snapshots)}) != volume count ({len(volumes)}); evaluating first {steps_to_eval} steps"
            )

        step_maps_strict: Dict[int, Dict[str, Any]] = {}
        step_maps_primary: Dict[int, Dict[str, Any]] = {}
        step_maps_pore: Dict[int, Dict[str, Any]] = {}
        gt_to_pore_group_map: Dict[int, int] = {}
        step_entries: List[Dict[str, Any]] = []

        for time_index in range(steps_to_eval):
            snapshot = snapshots[time_index]
            step_entry: Dict[str, Any] = {
                "time_index": int(time_index),
                "gt_annotations_path": None,
                "gt_labels_path": None,
                "evaluated": False,
                "errors": [],
                "warnings": [],
                "segmentation": {},
                "instance": {},
                "tracking": {},
                "mapping": {},
                "strict_instance": {},
                "strict_tracking": {},
                "strict_mapping": {},
                "alignment": {},
                "pore_level": {
                    "aggregation": {},
                    "instance": {},
                    "tracking": {},
                    "mapping": {},
                },
            }

            primary_eval: Optional[Dict[str, Any]] = None
            strict_eval: Optional[Dict[str, Any]] = None
            strict_pore_eval: Optional[Dict[str, Any]] = None

            # Primary channel: annotations.json pore-level evaluation.
            ann_payload, ann_path, ann_error = self._resolve_annotations_payload(volumes[time_index])
            step_entry["gt_annotations_path"] = ann_path
            if ann_error:
                step_entry["warnings"].append(ann_error)

            if isinstance(ann_payload, dict):
                try:
                    primary_eval = self._evaluate_step_against_annotations(
                        snapshot=snapshot,
                        annotations=ann_payload,
                        instance_iou_threshold=instance_iou_threshold,
                    )
                except Exception as exc:
                    step_entry["errors"].append(f"failed to evaluate annotations.json: {exc}")
            else:
                step_entry["warnings"].append(
                    "annotations.json missing in volume metadata "
                    "sim_annotations.annotations/files.annotations_json"
                )

            # Strict diagnostics channel: labels.npy voxel/instance evaluation.
            predicted_labels = snapshot.segmented_regions
            labels_path = self._resolve_labels_path(volumes[time_index])
            step_entry["gt_labels_path"] = labels_path
            if labels_path and predicted_labels is not None:
                try:
                    gt_labels = np.load(labels_path, mmap_mode="r")
                except Exception as exc:
                    step_entry["warnings"].append(f"failed to read labels.npy for strict diagnostics: {exc}")
                else:
                    alignment = infer_shape_alignment(
                        pred_shape=predicted_labels.shape,
                        gt_shape=gt_labels.shape,
                        min_overlap_ratio=0.85,
                    )
                    if alignment is None:
                        step_entry["warnings"].append(
                            "strict diagnostics skipped: shape mismatch not alignable "
                            f"(gt={tuple(gt_labels.shape)} vs pred={tuple(predicted_labels.shape)})"
                        )
                    else:
                        step_entry["alignment"] = alignment.to_dict()
                        if tuple(gt_labels.shape) != tuple(predicted_labels.shape):
                            step_entry["warnings"].append(
                                "shape mismatch resolved by axis/crop alignment "
                                f"(overlap_ratio={alignment.overlap_ratio:.3f})"
                            )

                        pred_eval, gt_eval = align_pred_gt(predicted_labels, gt_labels, alignment)
                        strict_eval = self._evaluate_step_against_gt(
                            predicted_labels=pred_eval,
                            gt_labels=gt_eval,
                            instance_iou_threshold=instance_iou_threshold,
                        )
                        if time_index == 0:
                            gt_to_pore_group_map = self._build_gt_to_pore_group_map(
                                predicted_labels_t0=pred_eval,
                                gt_labels_t0=gt_eval,
                            )
                        if gt_to_pore_group_map:
                            gt_pore_eval, agg_info = self._remap_gt_labels_with_group_map(
                                gt_labels=gt_eval,
                                gt_to_group=gt_to_pore_group_map,
                            )
                            strict_pore_eval = self._evaluate_step_against_gt(
                                predicted_labels=pred_eval,
                                gt_labels=gt_pore_eval,
                                instance_iou_threshold=instance_iou_threshold,
                            )
                            step_entry["pore_level"]["aggregation"] = {
                                **agg_info,
                                "t0_group_map_size": int(len(gt_to_pore_group_map)),
                            }
                            step_entry["pore_level"]["instance"] = strict_pore_eval["instance"]
                            step_entry["pore_level"]["mapping"] = strict_pore_eval["mapping"]
                            step_maps_pore[time_index] = strict_pore_eval["mapping"]
                        step_entry["segmentation"] = strict_eval["segmentation"]
                        step_entry["strict_instance"] = strict_eval["instance"]
                        step_entry["strict_mapping"] = strict_eval["mapping"]
                        step_maps_strict[time_index] = strict_eval["mapping"]
            elif labels_path and predicted_labels is None:
                step_entry["warnings"].append(
                    "snapshot has no segmented_regions; strict voxel-level diagnostics skipped"
                )

            # Fallback for legacy datasets without annotations.json.
            if primary_eval is None and strict_eval is not None:
                if strict_pore_eval is not None:
                    primary_eval = {
                        "segmentation": strict_eval.get("segmentation", {}),
                        "instance": strict_pore_eval.get("instance", {}),
                        "mapping": strict_pore_eval.get("mapping", {}),
                        "annotation": step_entry["pore_level"].get("aggregation", {}),
                    }
                else:
                    primary_eval = strict_eval
                step_entry["warnings"].append(
                    "primary evaluation fell back to labels.npy because annotations.json was unavailable"
                )

            if isinstance(primary_eval, dict):
                step_entry["evaluated"] = True
                if isinstance(primary_eval.get("segmentation"), dict) and primary_eval["segmentation"]:
                    step_entry["segmentation"] = primary_eval["segmentation"]
                step_entry["instance"] = primary_eval.get("instance", {})
                step_entry["mapping"] = primary_eval.get("mapping", {})
                step_maps_primary[time_index] = step_entry["mapping"]

                annotation_info = primary_eval.get("annotation", {})
                if isinstance(annotation_info, dict):
                    step_entry["pore_level"]["aggregation"] = annotation_info
                else:
                    step_entry["pore_level"]["aggregation"] = step_entry["pore_level"].get("aggregation", {})
                if not step_entry["pore_level"].get("instance"):
                    step_entry["pore_level"]["instance"] = dict(step_entry["instance"])
                if not step_entry["pore_level"].get("mapping"):
                    step_entry["pore_level"]["mapping"] = step_entry["mapping"]
                    step_maps_pore[time_index] = step_entry["mapping"]

            step_entries.append(step_entry)

        reference_map_strict = step_maps_strict.get(0, {}).get("pred_to_gt", {})
        reference_gt_ids_strict = set(reference_map_strict.values()) if isinstance(reference_map_strict, dict) else set()
        t0_gt_ids_strict = set(step_maps_strict.get(0, {}).get("gt_ids", set()))
        t0_reference_gt_coverage_strict = self._safe_ratio(
            len(reference_gt_ids_strict),
            len(t0_gt_ids_strict),
            default=1.0 if len(t0_gt_ids_strict) == 0 else 0.0,
        )

        reference_map_pore = step_maps_pore.get(0, {}).get("pred_to_gt", {})
        reference_gt_ids_pore = set(reference_map_pore.values()) if isinstance(reference_map_pore, dict) else set()
        t0_gt_ids_pore = set(step_maps_pore.get(0, {}).get("gt_ids", set()))
        t0_reference_gt_coverage_pore = self._safe_ratio(
            len(reference_gt_ids_pore),
            len(t0_gt_ids_pore),
            default=1.0 if len(t0_gt_ids_pore) == 0 else 0.0,
        )
        reference_map = step_maps_primary.get(0, {}).get("pred_to_gt", {})
        reference_gt_ids = set(reference_map.values()) if isinstance(reference_map, dict) else set()
        t0_gt_ids = set(step_maps_primary.get(0, {}).get("gt_ids", set()))
        t0_reference_gt_coverage = self._safe_ratio(
            len(reference_gt_ids),
            len(t0_gt_ids),
            default=1.0 if len(t0_gt_ids) == 0 else 0.0,
        )

        if not reference_map:
            report["warnings"].append(
                "step 0 has no usable pred->GT mapping in primary (annotations pore-level) evaluation; "
                "tracking identity accuracy will be skipped"
            )
        if not reference_map_pore and any(isinstance(row.get("pore_level"), dict) for row in step_entries):
            report["warnings"].append(
                "step 0 has no usable pred->GT mapping for pore-level evaluation; "
                "pore-level tracking accuracy will be skipped"
            )

        for time_index in range(1, steps_to_eval):
            step_entry = step_entries[time_index]
            if not step_entry.get("evaluated", False):
                continue
            step_map = step_maps_primary.get(time_index)
            if not isinstance(step_map, dict):
                continue

            id_map = self.time_series.tracking.id_mapping.get(time_index, {})
            novel_stats = snapshots[time_index].metadata.get("novel_segmentation", {})
            tracking_eval = self._evaluate_tracking_step(
                time_index=time_index,
                id_map=id_map if isinstance(id_map, dict) else {},
                reference_pred_to_gt=dict(reference_map) if isinstance(reference_map, dict) else {},
                current_pred_to_gt=dict(step_map.get("pred_to_gt", {})),
                current_gt_ids=set(step_map.get("gt_ids", set())),
                novel_segmentation=novel_stats if isinstance(novel_stats, dict) else {},
            )
            step_entry["tracking"] = tracking_eval

            step_map_strict = step_maps_strict.get(time_index)
            if isinstance(step_map_strict, dict) and isinstance(reference_map_strict, dict) and reference_map_strict:
                tracking_eval_strict = self._evaluate_tracking_step(
                    time_index=time_index,
                    id_map=id_map if isinstance(id_map, dict) else {},
                    reference_pred_to_gt=dict(reference_map_strict),
                    current_pred_to_gt=dict(step_map_strict.get("pred_to_gt", {})),
                    current_gt_ids=set(step_map_strict.get("gt_ids", set())),
                    novel_segmentation=novel_stats if isinstance(novel_stats, dict) else {},
                )
                step_entry["strict_tracking"] = tracking_eval_strict

            step_map_pore = step_maps_pore.get(time_index)
            if isinstance(step_map_pore, dict) and isinstance(reference_map_pore, dict) and reference_map_pore:
                tracking_eval_pore = self._evaluate_tracking_step(
                    time_index=time_index,
                    id_map=id_map if isinstance(id_map, dict) else {},
                    reference_pred_to_gt=dict(reference_map_pore),
                    current_pred_to_gt=dict(step_map_pore.get("pred_to_gt", {})),
                    current_gt_ids=set(step_map_pore.get("gt_ids", set())),
                    novel_segmentation=novel_stats if isinstance(novel_stats, dict) else {},
                )
                step_entry["pore_level"]["tracking"] = tracking_eval_pore

        evaluated_steps = [row for row in step_entries if row.get("evaluated", False)]
        evaluated_detection = [row for row in evaluated_steps if isinstance(row.get("instance"), dict) and row["instance"]]
        evaluated_detection_pore = [
            row
            for row in evaluated_steps
            if isinstance(row.get("pore_level"), dict)
            and isinstance(row["pore_level"].get("instance"), dict)
            and row["pore_level"]["instance"]
        ]
        evaluated_tracking = [
            row for row in step_entries[1:]
            if isinstance(row.get("tracking"), dict) and bool(row["tracking"].get("available", False))
        ]
        evaluated_tracking_strict = [
            row for row in step_entries[1:]
            if isinstance(row.get("strict_tracking"), dict) and bool(row["strict_tracking"].get("available", False))
        ]
        evaluated_tracking_pore = [
            row
            for row in step_entries[1:]
            if isinstance(row.get("pore_level"), dict)
            and isinstance(row["pore_level"].get("tracking"), dict)
            and bool(row["pore_level"]["tracking"].get("available", False))
        ]

        overall: Dict[str, Any] = {
            "num_steps_total": len(step_entries),
            "num_steps_evaluated": len(evaluated_steps),
            "num_steps_with_tracking_eval": len(evaluated_tracking),
            "num_steps_with_strict_tracking_eval": len(evaluated_tracking_strict),
            "num_steps_with_pore_level_tracking_eval": len(evaluated_tracking_pore),
            "t0_reference_gt_coverage": float(t0_reference_gt_coverage),
            "t0_reference_pred_to_gt_size": int(len(reference_map)) if isinstance(reference_map, dict) else 0,
            "t0_gt_instance_count": int(len(t0_gt_ids)),
            "t0_reference_gt_coverage_strict": float(t0_reference_gt_coverage_strict),
            "t0_reference_pred_to_gt_size_strict": int(len(reference_map_strict)) if isinstance(reference_map_strict, dict) else 0,
            "t0_gt_instance_count_strict": int(len(t0_gt_ids_strict)),
            "t0_pore_level_reference_gt_coverage": float(t0_reference_gt_coverage_pore),
            "t0_pore_level_reference_map_size": int(len(reference_map_pore)) if isinstance(reference_map_pore, dict) else 0,
            "t0_pore_level_gt_group_count": int(len(t0_gt_ids_pore)),
            # Legacy key retained for backward-compatible consumers.
            "t0_gt_to_pore_group_map_size": int(len(gt_to_pore_group_map)),
        }

        if evaluated_detection:
            mean_instance_precision = float(np.mean([row["instance"].get("precision", 0.0) for row in evaluated_detection]))
            mean_instance_recall = float(np.mean([row["instance"].get("recall", 0.0) for row in evaluated_detection]))
            mean_instance_f1 = float(np.mean([row["instance"].get("f1", 0.0) for row in evaluated_detection]))
            overall.update(
                {
                    "mean_instance_precision": mean_instance_precision,
                    "mean_instance_recall": mean_instance_recall,
                    "mean_instance_f1": mean_instance_f1,
                }
            )
            rows_with_voxel = [
                row
                for row in evaluated_detection
                if isinstance(row.get("segmentation"), dict) and "voxel_iou" in row["segmentation"]
            ]
            if rows_with_voxel:
                overall["mean_voxel_iou"] = float(
                    np.mean([row["segmentation"].get("voxel_iou", 0.0) for row in rows_with_voxel])
                )

        if evaluated_detection_pore:
            mean_pore_instance_precision = float(
                np.mean([row["pore_level"]["instance"].get("precision", 0.0) for row in evaluated_detection_pore])
            )
            mean_pore_instance_recall = float(
                np.mean([row["pore_level"]["instance"].get("recall", 0.0) for row in evaluated_detection_pore])
            )
            mean_pore_instance_f1 = float(
                np.mean([row["pore_level"]["instance"].get("f1", 0.0) for row in evaluated_detection_pore])
            )
            overall.update(
                {
                    "mean_pore_level_instance_precision": mean_pore_instance_precision,
                    "mean_pore_level_instance_recall": mean_pore_instance_recall,
                    "mean_pore_level_instance_f1": mean_pore_instance_f1,
                }
            )
        elif evaluated_detection:
            overall["mean_pore_level_instance_precision"] = float(overall.get("mean_instance_precision", 0.0))
            overall["mean_pore_level_instance_recall"] = float(overall.get("mean_instance_recall", 0.0))
            overall["mean_pore_level_instance_f1"] = float(overall.get("mean_instance_f1", 0.0))

        if evaluated_tracking:
            mean_tracking_accuracy = float(np.mean([row["tracking"].get("accuracy", 0.0) for row in evaluated_tracking]))
            mean_active_accuracy = float(np.mean([row["tracking"].get("active_accuracy", 0.0) for row in evaluated_tracking]))
            mean_closure_accuracy = float(np.mean([row["tracking"].get("closure_accuracy", 0.0) for row in evaluated_tracking]))
            mean_reference_scope_gt_coverage = float(
                np.mean([row["tracking"].get("reference_scope_gt_coverage", 0.0) for row in evaluated_tracking])
            )
            mean_untracked_novel_segments = float(
                np.mean([row["tracking"].get("untracked_novel_segments", 0.0) for row in evaluated_tracking])
            )
            overall.update(
                {
                    "mean_tracking_accuracy": mean_tracking_accuracy,
                    "mean_active_tracking_accuracy": mean_active_accuracy,
                    "mean_closure_accuracy": mean_closure_accuracy,
                    "mean_reference_scope_gt_coverage": mean_reference_scope_gt_coverage,
                    "mean_untracked_novel_segments": mean_untracked_novel_segments,
                }
            )

        if evaluated_tracking_strict:
            overall.update(
                {
                    "mean_tracking_accuracy_strict": float(
                        np.mean([row["strict_tracking"].get("accuracy", 0.0) for row in evaluated_tracking_strict])
                    ),
                    "mean_active_tracking_accuracy_strict": float(
                        np.mean([row["strict_tracking"].get("active_accuracy", 0.0) for row in evaluated_tracking_strict])
                    ),
                    "mean_closure_accuracy_strict": float(
                        np.mean([row["strict_tracking"].get("closure_accuracy", 0.0) for row in evaluated_tracking_strict])
                    ),
                }
            )

        if evaluated_tracking_pore:
            mean_tracking_accuracy_pore = float(
                np.mean([row["pore_level"]["tracking"].get("accuracy", 0.0) for row in evaluated_tracking_pore])
            )
            mean_active_accuracy_pore = float(
                np.mean([row["pore_level"]["tracking"].get("active_accuracy", 0.0) for row in evaluated_tracking_pore])
            )
            mean_closure_accuracy_pore = float(
                np.mean([row["pore_level"]["tracking"].get("closure_accuracy", 0.0) for row in evaluated_tracking_pore])
            )
            mean_reference_scope_gt_coverage_pore = float(
                np.mean(
                    [
                        row["pore_level"]["tracking"].get("reference_scope_gt_coverage", 0.0)
                        for row in evaluated_tracking_pore
                    ]
                )
            )
            overall.update(
                {
                    "mean_pore_level_tracking_accuracy": mean_tracking_accuracy_pore,
                    "mean_pore_level_active_tracking_accuracy": mean_active_accuracy_pore,
                    "mean_pore_level_closure_accuracy": mean_closure_accuracy_pore,
                    "mean_pore_level_reference_scope_gt_coverage": mean_reference_scope_gt_coverage_pore,
                }
            )
            # Keep primary top-level tracking metrics aligned with pore-level policy.
            overall["mean_tracking_accuracy"] = mean_tracking_accuracy_pore
            overall["mean_active_tracking_accuracy"] = mean_active_accuracy_pore
            overall["mean_closure_accuracy"] = mean_closure_accuracy_pore
            overall["mean_reference_scope_gt_coverage"] = mean_reference_scope_gt_coverage_pore
        elif evaluated_tracking:
            overall["mean_pore_level_tracking_accuracy"] = float(overall.get("mean_tracking_accuracy", 0.0))
            overall["mean_pore_level_active_tracking_accuracy"] = float(
                overall.get("mean_active_tracking_accuracy", 0.0)
            )
            overall["mean_pore_level_closure_accuracy"] = float(
                overall.get("mean_closure_accuracy", 0.0)
            )
            overall["mean_pore_level_reference_scope_gt_coverage"] = float(
                overall.get("mean_reference_scope_gt_coverage", 0.0)
            )

        for step_entry in step_entries:
            for msg in step_entry.get("errors", []):
                report["errors"].append(f"t={step_entry['time_index']} {msg}")
            for msg in step_entry.get("warnings", []):
                report["warnings"].append(f"t={step_entry['time_index']} {msg}")

        report["steps"] = step_entries
        report["overall"] = overall
        report["available"] = bool(evaluated_steps)
        if report["errors"]:
            report["status"] = "partial" if report["available"] else "unavailable"
        else:
            report["status"] = "ok" if report["available"] else "unavailable"

        self.time_series.tracking.evaluation = report
        return report

    def get_results(self) -> TimeSeriesPNM:
        return self.time_series

    def get_volume_history(self, pore_id: int) -> List[float]:
        return self.time_series.tracking.get_volume_series(pore_id)

    def get_pore_status(self, pore_id: int, timepoint: int) -> PoreStatus:
        statuses = self.time_series.tracking.status_history.get(pore_id, [])
        if timepoint < len(statuses):
            return statuses[timepoint]
        return PoreStatus.UNKNOWN

    def export_volume_csv(self, filepath: str) -> None:
        import csv

        tracking = self.time_series.tracking
        num_timepoints = self.time_series.num_timepoints

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["pore_id"] + [f"t{i}" for i in range(num_timepoints)] + ["final_status", "volume_retention"]
            writer.writerow(header)

            for pore_id in tracking.reference_ids:
                volumes = tracking.volume_history.get(pore_id, [])
                statuses = tracking.status_history.get(pore_id, [])
                final_status = statuses[-1].value if statuses else "unknown"
                if volumes and volumes[0] > 0:
                    retention = volumes[-1] / volumes[0]
                else:
                    retention = 0.0
                row = [pore_id] + volumes + [final_status, f"{retention:.4f}"]
                writer.writerow(row)

        print(f"[Tracker] Exported volume history to {filepath}")


# ==========================================
# Batch IoU Calculation Functions
# ==========================================

def compute_batch_iou_cpu(
    reference_masks: List[np.ndarray],
    current_labels: np.ndarray,
    ref_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IoU scores for all reference pores in batch (CPU version).
    """
    num_pores = len(reference_masks)
    matched_ids = np.zeros(num_pores, dtype=np.int32)
    iou_scores = np.zeros(num_pores, dtype=np.float32)
    volumes = np.zeros(num_pores, dtype=np.float32)

    for i, mask_data in enumerate(reference_masks):
        bbox = mask_data["bbox"]
        local_mask = mask_data["mask"]
        mins, maxs = bbox
        current_region = current_labels[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        overlapping = current_region[local_mask]
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 0:
            continue

        counts = np.bincount(overlapping)
        if len(counts) > 1:
            best_label = int(np.argmax(counts[1:]) + 1)
        else:
            best_label = int(overlapping[0])

        current_mask = current_region == best_label
        intersection = np.sum(local_mask & current_mask)
        union = np.sum(local_mask | current_mask)

        if union > 0:
            matched_ids[i] = best_label
            iou_scores[i] = intersection / union
            volumes[i] = np.sum(current_mask)

    return matched_ids, iou_scores, volumes


def compute_batch_iou_gpu(
    reference_masks: List[Dict],
    current_labels: np.ndarray,
    ref_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IoU scores for all reference pores in batch (GPU version).
    """
    if not HAS_GPU or cp is None:
        raise RuntimeError("GPU acceleration requested but CuPy not available")

    current_labels_gpu = cp.asarray(current_labels)
    num_pores = len(reference_masks)
    matched_ids = cp.zeros(num_pores, dtype=cp.int32)
    iou_scores = cp.zeros(num_pores, dtype=cp.float32)
    volumes = cp.zeros(num_pores, dtype=cp.float32)

    for i, mask_data in enumerate(reference_masks):
        bbox = mask_data["bbox"]
        local_mask_cpu = mask_data["mask"]
        mins, maxs = bbox

        current_region = current_labels_gpu[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        local_mask_gpu = cp.asarray(local_mask_cpu)
        overlapping = current_region[local_mask_gpu]
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 0:
            continue

        counts = cp.bincount(overlapping)
        if len(counts) > 1:
            best_label = int(cp.argmax(counts[1:]) + 1)
        else:
            best_label = int(overlapping[0])

        current_mask = current_region == best_label
        intersection = cp.sum(local_mask_gpu & current_mask)
        union = cp.sum(local_mask_gpu | current_mask)
        if union > 0:
            matched_ids[i] = best_label
            iou_scores[i] = intersection / union
            volumes[i] = cp.sum(current_mask)

    return cp.asnumpy(matched_ids), cp.asnumpy(iou_scores), cp.asnumpy(volumes)


def match_with_hungarian(
    iou_matrix: np.ndarray,
    iou_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform global assignment from IoU matrix using scipy Hungarian solver.
    """
    if not HAS_SCIPY:
        raise RuntimeError("Hungarian algorithm requested but scipy not available")

    if iou_matrix.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix < iou_threshold] = INVALID_COST
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    return row_ind[valid].astype(np.int32), col_ind[valid].astype(np.int32)
