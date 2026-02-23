"""
Simulation annotation parsing and validation service for 4D CT series.

This module intentionally keeps annotation logic outside data ingestion
components to preserve single responsibility in loaders.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from core import VolumeData
from core.annotation_alignment import infer_shape_alignment


class AnnotationValidator:
    """
    Validate per-step simulation annotations against loaded VolumeData objects.
    """

    def __init__(self, strict: bool = False):
        self.strict = bool(strict)

    def validate_series(
        self,
        parent_folder: str,
        volumes: List[VolumeData],
    ) -> Dict[str, Any]:
        """
        Parse and validate series-level + step-level annotation assets.

        Side effect:
            Enriches each volume metadata with:
            - sim_annotations
            - sim_annotation_series
        """
        summary_bundle: Dict[str, Any] = self._load_series_annotations_summary(
            parent_folder=parent_folder,
            expected_steps=len(volumes),
        )

        series_validation_errors: List[str] = []
        series_validation_warnings: List[str] = []

        for i, volume in enumerate(volumes):
            if not isinstance(volume.metadata, dict):
                volume.metadata = {}

            step_folder = volume.metadata.get("source_folder")
            if not isinstance(step_folder, str) or not step_folder:
                step_folder = ""

            step_annotation = self._load_step_annotations(
                step_folder=step_folder,
                volume=volume,
                step_index=i,
                summary_bundle=summary_bundle,
            )
            volume.metadata["sim_annotations"] = step_annotation

            if step_annotation["validation"]["errors"]:
                for item in step_annotation["validation"]["errors"]:
                    series_validation_errors.append(f"t={i} {item}")
            if step_annotation["validation"]["warnings"]:
                for item in step_annotation["validation"]["warnings"]:
                    series_validation_warnings.append(f"t={i} {item}")

        summary_validation = summary_bundle.get("validation", {})
        for msg in summary_validation.get("errors", []):
            series_validation_errors.append(f"summary {msg}")
        for msg in summary_validation.get("warnings", []):
            series_validation_warnings.append(f"summary {msg}")

        series_annotation_report = {
            "summary": summary_bundle,
            "validation": {
                "ok": len(series_validation_errors) == 0,
                "errors": series_validation_errors,
                "warnings": series_validation_warnings,
                "error_count": len(series_validation_errors),
                "warning_count": len(series_validation_warnings),
            },
        }

        for volume in volumes:
            if not isinstance(volume.metadata, dict):
                volume.metadata = {}
            volume.metadata["sim_annotation_series"] = series_annotation_report

        if self.strict and series_validation_errors:
            raise ValueError(
                "Simulation annotation validation failed: "
                + "; ".join(series_validation_errors[:5])
            )

        return series_annotation_report

    def _load_series_annotations_summary(self, parent_folder: str, expected_steps: int) -> Dict[str, Any]:
        """Load and lightly validate annotations_summary.json at series root."""
        summary_path = os.path.join(parent_folder, "annotations_summary.json")
        report: Dict[str, Any] = {
            "available": False,
            "path": summary_path,
            "data": None,
            "steps_by_index": {},
            "validation": {"ok": True, "errors": [], "warnings": []},
        }
        if not os.path.isfile(summary_path):
            report["validation"]["warnings"].append("annotations_summary.json not found")
            return report

        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception as exc:
            report["validation"]["ok"] = False
            report["validation"]["errors"].append(f"Failed to parse annotations_summary.json: {exc}")
            return report

        report["available"] = True
        report["data"] = summary

        steps = summary.get("steps", [])
        if not isinstance(steps, list):
            steps = []
            report["validation"]["errors"].append("annotations_summary.json field 'steps' is not a list")

        for row in steps:
            if not isinstance(row, dict):
                continue
            idx = row.get("step_index")
            if isinstance(idx, int):
                report["steps_by_index"][idx] = row

        num_steps_declared = summary.get("num_steps")
        if isinstance(num_steps_declared, int) and num_steps_declared != expected_steps:
            report["validation"]["warnings"].append(
                f"summary num_steps={num_steps_declared} differs from loaded timepoints={expected_steps}"
            )
        if len(report["steps_by_index"]) and len(report["steps_by_index"]) != expected_steps:
            report["validation"]["warnings"].append(
                f"summary steps entries={len(report['steps_by_index'])} differs from loaded timepoints={expected_steps}"
            )

        if report["validation"]["errors"]:
            report["validation"]["ok"] = False
        return report

    def _load_step_annotations(
        self,
        step_folder: str,
        volume: VolumeData,
        step_index: int,
        summary_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load per-step simulation annotations and validate against CT volume."""
        annotations_path = os.path.join(step_folder, "annotations.json")
        coco_path = os.path.join(step_folder, "coco_annotations.json")
        labels_path = os.path.join(step_folder, "labels.npy")

        step_report: Dict[str, Any] = {
            "step_index": int(step_index),
            "folder": step_folder,
            "files": {
                "annotations_json": annotations_path if os.path.isfile(annotations_path) else None,
                "coco_annotations_json": coco_path if os.path.isfile(coco_path) else None,
                "labels_npy": labels_path if os.path.isfile(labels_path) else None,
            },
            "annotations": None,
            "coco": None,
            "labels": None,
            "validation": {"ok": True, "errors": [], "warnings": [], "checks": {}},
        }

        if step_report["files"]["annotations_json"] is not None:
            try:
                with open(annotations_path, "r", encoding="utf-8") as fh:
                    step_report["annotations"] = json.load(fh)
            except Exception as exc:
                step_report["validation"]["errors"].append(f"Failed to parse annotations.json: {exc}")
        else:
            step_report["validation"]["warnings"].append("annotations.json missing")

        if step_report["files"]["coco_annotations_json"] is not None:
            try:
                with open(coco_path, "r", encoding="utf-8") as fh:
                    step_report["coco"] = json.load(fh)
            except Exception as exc:
                step_report["validation"]["errors"].append(f"Failed to parse coco_annotations.json: {exc}")
        else:
            step_report["validation"]["warnings"].append("coco_annotations.json missing")

        if step_report["files"]["labels_npy"] is not None:
            try:
                labels_volume = np.load(labels_path, mmap_mode="r")
                step_report["labels"] = self._collect_label_stats(labels_volume)
            except Exception as exc:
                step_report["validation"]["errors"].append(f"Failed to load labels.npy: {exc}")
        else:
            step_report["validation"]["warnings"].append("labels.npy missing")

        summary_step = summary_bundle.get("steps_by_index", {}).get(step_index)
        validation = self._validate_step_annotations(
            step_index=step_index,
            volume_shape=list(volume.dimensions),
            annotations=step_report["annotations"],
            coco=step_report["coco"],
            label_stats=step_report["labels"],
            summary_step=summary_step,
            summary_available=bool(summary_bundle.get("available", False)),
        )
        step_report["validation"] = validation
        return step_report

    def _collect_label_stats(self, labels_volume: np.ndarray) -> Dict[str, Any]:
        """Collect labels.npy stats without forcing full-memory copy."""
        stats: Dict[str, Any] = {
            "shape": [int(v) for v in labels_volume.shape],
            "dtype": str(labels_volume.dtype),
            "unique_ids": [],
            "unique_ids_truncated": False,
            "nonzero_count": 0,
            "nonzero_count_estimated": False,
        }

        full_scan_voxel_limit = 40_000_000
        if labels_volume.size <= full_scan_voxel_limit:
            unique_all = np.unique(labels_volume)
            unique_positive = unique_all[unique_all > 0]
            stats["unique_ids"] = [int(v) for v in unique_positive.tolist()]
            stats["nonzero_count"] = int(np.count_nonzero(labels_volume))
            return stats

        flat = labels_volume.reshape(-1)
        stride = max(1, int(flat.size // 5_000_000))
        sample = flat[::stride]
        unique_sample = np.unique(sample)
        unique_positive = unique_sample[unique_sample > 0]
        stats["unique_ids"] = [int(v) for v in unique_positive.tolist()]
        stats["unique_ids_truncated"] = True
        sampled_nonzero = int(np.count_nonzero(sample))
        stats["nonzero_count"] = int(sampled_nonzero * stride)
        stats["nonzero_count_estimated"] = True
        return stats

    def _validate_step_annotations(
        self,
        step_index: int,
        volume_shape: List[int],
        annotations: Optional[Dict[str, Any]],
        coco: Optional[Dict[str, Any]],
        label_stats: Optional[Dict[str, Any]],
        summary_step: Optional[Dict[str, Any]],
        summary_available: bool,
    ) -> Dict[str, Any]:
        """Validate one step's annotation assets and return structured report."""
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Any] = {}

        annotation_ids = set()
        if annotations is not None:
            ann_step_idx = annotations.get("step_index")
            if isinstance(ann_step_idx, int):
                checks["step_index_match"] = ann_step_idx == step_index
                if ann_step_idx != step_index:
                    errors.append(f"annotations step_index={ann_step_idx} != loaded index={step_index}")
            else:
                warnings.append("annotations step_index missing or not int")

            ann_shape = annotations.get("volume_shape")
            if isinstance(ann_shape, list) and len(ann_shape) == 3:
                ann_shape_i = [int(v) for v in ann_shape]
                checks["volume_shape_match"] = ann_shape_i == volume_shape
                if ann_shape_i != volume_shape:
                    alignment = infer_shape_alignment(volume_shape, ann_shape_i)
                    if alignment is None:
                        checks["volume_shape_alignable"] = False
                        errors.append(f"annotations volume_shape={ann_shape_i} != CT shape={volume_shape}")
                    else:
                        checks["volume_shape_alignable"] = True
                        checks["volume_shape_alignment"] = alignment.to_dict()
                        if alignment.overlap_ratio < 1.0:
                            warnings.append(
                                "annotations volume_shape differs from CT; "
                                "evaluation will center-crop to overlap region"
                            )
                        else:
                            warnings.append(
                                "annotations volume_shape differs from CT axis order; "
                                "evaluation will apply axis alignment"
                            )
            else:
                warnings.append("annotations volume_shape missing or invalid")

            voids = annotations.get("voids", [])
            if not isinstance(voids, list):
                warnings.append("annotations voids missing or not list")
                voids = []
            parsed_ids = []
            for item in voids:
                if not isinstance(item, dict):
                    continue
                vid = item.get("id")
                if isinstance(vid, int) and vid > 0:
                    parsed_ids.append(vid)
            annotation_ids = set(parsed_ids)
            checks["num_voids_detected"] = len(annotation_ids)
            if len(parsed_ids) != len(annotation_ids):
                errors.append("annotations void ids are duplicated")

            num_voids = annotations.get("num_voids")
            if isinstance(num_voids, int):
                checks["num_voids_match"] = (num_voids == len(voids))
                if num_voids != len(voids):
                    warnings.append(f"annotations num_voids={num_voids} but len(voids)={len(voids)}")
            else:
                warnings.append("annotations num_voids missing or not int")
        else:
            warnings.append("annotations.json unavailable, skipped detailed 3D checks")

        if coco is not None:
            for required in ("images", "annotations", "categories"):
                if required not in coco:
                    warnings.append(f"coco_annotations.json missing key '{required}'")
            checks["coco_annotations_count"] = (
                len(coco.get("annotations", [])) if isinstance(coco.get("annotations"), list) else 0
            )
        else:
            warnings.append("coco_annotations.json unavailable, skipped COCO checks")

        if label_stats is not None:
            label_shape = label_stats.get("shape")
            if isinstance(label_shape, list) and len(label_shape) == 3:
                label_shape_i = [int(v) for v in label_shape]
                checks["labels_shape_match"] = label_shape_i == [int(v) for v in volume_shape]
                if not checks["labels_shape_match"]:
                    alignment = infer_shape_alignment(volume_shape, label_shape_i)
                    if alignment is None:
                        checks["labels_shape_alignable"] = False
                        errors.append(f"labels.npy shape={label_shape} != CT shape={volume_shape}")
                    else:
                        checks["labels_shape_alignable"] = True
                        checks["labels_shape_alignment"] = alignment.to_dict()
                        if alignment.overlap_ratio < 1.0:
                            warnings.append(
                                "labels.npy shape differs from CT; "
                                "evaluation will center-crop to overlap region"
                            )
                        else:
                            warnings.append(
                                "labels.npy shape differs from CT axis order; "
                                "evaluation will apply axis alignment"
                            )
            else:
                errors.append("labels.npy shape metadata missing")

            dtype_str = str(label_stats.get("dtype", ""))
            if "int" not in dtype_str:
                warnings.append(f"labels.npy dtype is '{dtype_str}', expected integer type")

            label_ids = set(int(v) for v in label_stats.get("unique_ids", []))
            checks["label_instance_count"] = len(label_ids)
            if annotation_ids:
                missing_ids = sorted(annotation_ids - label_ids)
                extra_ids = sorted(label_ids - annotation_ids)
                if missing_ids:
                    errors.append(f"labels.npy missing instance ids from annotations: {missing_ids[:20]}")
                if extra_ids:
                    warnings.append(f"labels.npy has extra instance ids not in annotations: {extra_ids[:20]}")
                if label_stats.get("unique_ids_truncated", False):
                    warnings.append("labels.npy unique id scan was sampled (very large volume)")
        else:
            warnings.append("labels.npy unavailable, skipped instance-volume checks")

        if summary_step is not None:
            step_in_summary = summary_step.get("step_index")
            if isinstance(step_in_summary, int) and step_in_summary != step_index:
                errors.append(f"summary step_index={step_in_summary} != loaded index={step_index}")

            if annotations is not None:
                ann_ratio = annotations.get("compression_ratio")
                sum_ratio = summary_step.get("compression_ratio")
                if isinstance(ann_ratio, (int, float)) and isinstance(sum_ratio, (int, float)):
                    if abs(float(ann_ratio) - float(sum_ratio)) > 1e-6:
                        warnings.append(
                            "compression_ratio mismatch between annotations.json "
                            f"({ann_ratio}) and summary ({sum_ratio})"
                        )

                ann_num_voids = annotations.get("num_voids")
                sum_num_voids = summary_step.get("num_voids")
                if isinstance(ann_num_voids, int) and isinstance(sum_num_voids, int):
                    if ann_num_voids != sum_num_voids:
                        warnings.append(
                            f"num_voids mismatch between annotations.json ({ann_num_voids}) "
                            f"and summary ({sum_num_voids})"
                        )
        elif summary_available and summary_step is None:
            warnings.append("annotations_summary has no entry for this step")

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

