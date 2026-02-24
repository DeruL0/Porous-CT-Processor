"""
Evaluation mixin extracted from pnm_tracker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.annotation_alignment import align_pred_gt, infer_shape_alignment
from processors.tracking.algorithms import INVALID_COST, solve_global_assignment


class PNMTrackerEvaluationMixin:
    """Evaluation-related methods shared by PNMTracker."""

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
        Evaluate segmentation + tracking quality against per-step simulation labels.npy.

        The method is non-destructive: missing annotation files produce warnings
        instead of raising, and partial reports are still returned.
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

            predicted_labels = snapshot.segmented_regions
            if predicted_labels is None:
                step_entry["errors"].append("snapshot has no segmented_regions")
                step_entries.append(step_entry)
                continue

            labels_path = self._resolve_labels_path(volumes[time_index])
            step_entry["gt_labels_path"] = labels_path
            if not labels_path:
                step_entry["warnings"].append("labels.npy path missing in volume metadata sim_annotations.files.labels_npy")
                step_entries.append(step_entry)
                continue

            try:
                gt_labels = np.load(labels_path, mmap_mode="r")
            except Exception as exc:
                step_entry["errors"].append(f"failed to read labels.npy: {exc}")
                step_entries.append(step_entry)
                continue

            alignment = infer_shape_alignment(
                pred_shape=predicted_labels.shape,
                gt_shape=gt_labels.shape,
                min_overlap_ratio=0.85,
            )
            if alignment is None:
                step_entry["errors"].append(
                    f"shape mismatch not alignable: gt={tuple(gt_labels.shape)} vs pred={tuple(predicted_labels.shape)}"
                )
                step_entries.append(step_entry)
                continue
            step_entry["alignment"] = alignment.to_dict()

            if tuple(gt_labels.shape) != tuple(predicted_labels.shape):
                step_entry["warnings"].append(
                    "shape mismatch resolved by axis/crop alignment "
                    f"(overlap_ratio={alignment.overlap_ratio:.3f})"
                )

            pred_eval, gt_eval = align_pred_gt(predicted_labels, gt_labels, alignment)

            eval_step = self._evaluate_step_against_gt(
                predicted_labels=pred_eval,
                gt_labels=gt_eval,
                instance_iou_threshold=instance_iou_threshold,
            )
            step_entry["evaluated"] = True
            step_entry["segmentation"] = eval_step["segmentation"]
            step_entry["strict_instance"] = eval_step["instance"]
            step_entry["strict_mapping"] = eval_step["mapping"]
            step_maps_strict[time_index] = eval_step["mapping"]

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
                eval_step_pore = self._evaluate_step_against_gt(
                    predicted_labels=pred_eval,
                    gt_labels=gt_pore_eval,
                    instance_iou_threshold=instance_iou_threshold,
                )
                step_entry["pore_level"]["aggregation"] = {
                    **agg_info,
                    "t0_group_map_size": int(len(gt_to_pore_group_map)),
                }
                step_entry["pore_level"]["instance"] = eval_step_pore["instance"]
                step_entry["pore_level"]["mapping"] = eval_step_pore["mapping"]
                step_maps_pore[time_index] = eval_step_pore["mapping"]
                # Primary evaluation channel uses pore-level merged GT.
                step_entry["instance"] = eval_step_pore["instance"]
                step_entry["mapping"] = eval_step_pore["mapping"]
                step_maps_primary[time_index] = eval_step_pore["mapping"]
            else:
                step_entry["pore_level"]["aggregation"] = {
                    "known_gt_ids": 0,
                    "unknown_gt_ids": 0,
                    "num_group_ids": 0,
                    "t0_group_map_size": 0,
                }
                # Fallback to strict mapping if pore-level merge is unavailable.
                step_entry["instance"] = eval_step["instance"]
                step_entry["mapping"] = eval_step["mapping"]
                step_maps_primary[time_index] = eval_step["mapping"]

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
                "step 0 has no usable pred->GT mapping in primary (pore-level) evaluation; "
                "tracking identity accuracy will be skipped"
            )
        if gt_to_pore_group_map and not reference_map_pore:
            report["warnings"].append(
                "step 0 has no usable pred->GT mapping for pore-level merged GT; "
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
            "t0_gt_to_pore_group_map_size": int(len(gt_to_pore_group_map)),
        }

        if evaluated_detection:
            mean_instance_precision = float(np.mean([row["instance"].get("precision", 0.0) for row in evaluated_detection]))
            mean_instance_recall = float(np.mean([row["instance"].get("recall", 0.0) for row in evaluated_detection]))
            mean_instance_f1 = float(np.mean([row["instance"].get("f1", 0.0) for row in evaluated_detection]))
            mean_voxel_iou = float(np.mean([row["segmentation"].get("voxel_iou", 0.0) for row in evaluated_detection]))
            overall.update(
                {
                    "mean_instance_precision": mean_instance_precision,
                    "mean_instance_recall": mean_instance_recall,
                    "mean_instance_f1": mean_instance_f1,
                    "mean_voxel_iou": mean_voxel_iou,
                }
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

