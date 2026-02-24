"""
Pore Network Model Tracker for 4D CT analysis.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.coordinates import (
    voxel_delta_zyx_to_world_delta_xyz,
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
from processors.pnm_tracker_eval import PNMTrackerEvaluationMixin
from processors.tracking.algorithms import (
    HAS_LAPJV,
    HAS_SCIPY,
    INVALID_COST,
    build_candidates,
    solve_global_assignment,
    update_tracks_with_hysteresis,
)
from processors.tracking_utils import (
    MacroRegistrationResult,
    ConstantAccelerationKalman3D,
    estimate_macro_registration,
    should_mark_closed_by_compression,
)

# Try to import GPU backend
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

class PNMTracker(PNMTrackerEvaluationMixin):
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
    Perform global assignment from IoU matrix using assignment solver path.
    """
    if iou_matrix.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix < iou_threshold] = INVALID_COST
    assignments, _solver_used = solve_global_assignment(
        cost_matrix=cost_matrix,
        assign_solver="scipy",
        invalid_cost=INVALID_COST,
    )
    if not assignments:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    rows = np.asarray([int(item[0]) for item in assignments], dtype=np.int32)
    cols = np.asarray([int(item[1]) for item in assignments], dtype=np.int32)
    return rows, cols
