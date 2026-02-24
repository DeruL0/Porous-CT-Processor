"""
Shared pore segmentation core used by both 3D extraction and 4D snapshot paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max

from config import (
    MIN_PEAK_DISTANCE,
    SEGMENTATION_FORCE_SEED_PER_COMPONENT,
    SEGMENTATION_MAX_SEEDS_PER_COMPONENT,
    SEGMENTATION_NECK_EROSION_ITERS,
    SEGMENTATION_PROFILE_DEFAULT,
    SEGMENTATION_SMALL_COMPONENT_VOXELS,
    SEGMENTATION_SPLIT_MODE_DEFAULT,
    SEGMENTATION_SUPPLEMENT_MARGIN,
    SEGMENTATION_SUPPLEMENT_QUANTILE,
)
from processors.utils import binary_fill_holes, distance_transform_edt, watershed_gpu


SegmentationProfile = Literal["legacy", "dual_adaptive"]
SplitMode = Literal["conservative", "balanced", "aggressive"]


@dataclass(frozen=True)
class PoreSegmentationConfig:
    profile: SegmentationProfile = SEGMENTATION_PROFILE_DEFAULT
    split_mode: SplitMode = SEGMENTATION_SPLIT_MODE_DEFAULT
    base_min_peak_distance: int = MIN_PEAK_DISTANCE
    supplement_quantile: float = SEGMENTATION_SUPPLEMENT_QUANTILE
    supplement_threshold_margin: float = SEGMENTATION_SUPPLEMENT_MARGIN
    small_component_voxels: int = SEGMENTATION_SMALL_COMPONENT_VOXELS
    max_seeds_per_component: int = SEGMENTATION_MAX_SEEDS_PER_COMPONENT
    force_seed_per_component: bool = SEGMENTATION_FORCE_SEED_PER_COMPONENT
    neck_erosion_iters: int = SEGMENTATION_NECK_EROSION_ITERS


@dataclass
class PoreSegmentationResult:
    pores_mask: np.ndarray
    distance_map: np.ndarray
    labels: np.ndarray
    num_pores: int
    debug: Dict[str, Any]


_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)
_SPLIT_SCALE = {
    "conservative": 0.50,
    "balanced": 0.35,
    "aggressive": 0.20,
}
_PEAK_KEEP_RATIO = {
    "conservative": 0.60,
    "balanced": 0.50,
    "aggressive": 0.35,
}


def segment_pores_from_raw(
    raw: np.ndarray,
    threshold: float,
    spacing: Tuple[float, float, float],
    config: Optional[PoreSegmentationConfig] = None,
) -> PoreSegmentationResult:
    """
    Segment pores from a raw 3D CT volume.

    Returns pores mask, EDT map, watershed labels, and diagnostics.
    """
    raw_arr = np.asarray(raw)
    if raw_arr.ndim != 3:
        raise ValueError("segment_pores_from_raw expects a 3D array.")

    cfg = config or PoreSegmentationConfig()
    debug: Dict[str, Any] = {
        "profile": cfg.profile,
        "split_mode": cfg.split_mode,
        "threshold": float(threshold),
    }

    if cfg.profile == "legacy":
        pores_mask = _build_legacy_mask(raw_arr, threshold)
        debug["supplement_enabled"] = False
        debug["supplement_quantile_value"] = None
        debug["supplement_voxels"] = 0
        debug["cavity_voxels"] = int(np.count_nonzero(pores_mask))
    else:
        pores_mask, dual_debug = _build_dual_adaptive_mask(raw_arr, threshold, cfg)
        pores_mask = _remove_small_components(pores_mask, min_size=2)
        debug.update(dual_debug)

    debug["pores_mask_voxels"] = int(np.count_nonzero(pores_mask))
    if not np.any(pores_mask):
        empty_shape = raw_arr.shape
        debug.update(
            {
                "num_components": 0,
                "num_forced_seeds": 0,
                "num_neck_split_seeds": 0,
                "num_markers": 0,
            }
        )
        return PoreSegmentationResult(
            pores_mask=pores_mask.astype(bool, copy=False),
            distance_map=np.zeros(empty_shape, dtype=np.float32),
            labels=np.zeros(empty_shape, dtype=np.int32),
            num_pores=0,
            debug=debug,
        )

    sampling = _spacing_xyz_to_sampling_zyx(spacing)
    distance_map = distance_transform_edt(pores_mask, sampling=sampling).astype(np.float32, copy=False)
    markers, marker_debug = _build_adaptive_markers(distance_map=distance_map, pores_mask=pores_mask, cfg=cfg)
    debug.update(marker_debug)

    labels = watershed_gpu(-distance_map, markers, mask=distance_map > 0).astype(np.int32, copy=False)
    num_pores = int(np.max(labels)) if labels.size else 0
    debug["num_pores"] = num_pores

    return PoreSegmentationResult(
        pores_mask=pores_mask.astype(bool, copy=False),
        distance_map=distance_map,
        labels=labels,
        num_pores=num_pores,
        debug=debug,
    )


def segment_pores_from_mask(
    pores_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    config: Optional[PoreSegmentationConfig] = None,
) -> PoreSegmentationResult:
    """
    Segment pores from an already-prepared binary pores mask.
    """
    mask = np.asarray(pores_mask).astype(bool, copy=False)
    if mask.ndim != 3:
        raise ValueError("segment_pores_from_mask expects a 3D mask.")

    cfg = config or PoreSegmentationConfig()
    debug: Dict[str, Any] = {
        "profile": cfg.profile,
        "split_mode": cfg.split_mode,
        "mask_source": "precomputed",
        "pores_mask_voxels": int(np.count_nonzero(mask)),
        "supplement_enabled": False,
        "supplement_quantile_value": None,
        "supplement_voxels": 0,
    }

    if not np.any(mask):
        empty_shape = mask.shape
        debug.update(
            {
                "num_components": 0,
                "num_forced_seeds": 0,
                "num_neck_split_seeds": 0,
                "num_markers": 0,
            }
        )
        return PoreSegmentationResult(
            pores_mask=mask,
            distance_map=np.zeros(empty_shape, dtype=np.float32),
            labels=np.zeros(empty_shape, dtype=np.int32),
            num_pores=0,
            debug=debug,
        )

    sampling = _spacing_xyz_to_sampling_zyx(spacing)
    distance_map = distance_transform_edt(mask, sampling=sampling).astype(np.float32, copy=False)
    markers, marker_debug = _build_adaptive_markers(distance_map=distance_map, pores_mask=mask, cfg=cfg)
    debug.update(marker_debug)

    labels = watershed_gpu(-distance_map, markers, mask=distance_map > 0).astype(np.int32, copy=False)
    num_pores = int(np.max(labels)) if labels.size else 0
    debug["num_pores"] = num_pores

    return PoreSegmentationResult(
        pores_mask=mask,
        distance_map=distance_map,
        labels=labels,
        num_pores=num_pores,
        debug=debug,
    )


def _build_legacy_mask(raw: np.ndarray, threshold: float) -> np.ndarray:
    solid_mask = raw > threshold
    filled_solid = binary_fill_holes(solid_mask)
    return (filled_solid ^ solid_mask).astype(bool, copy=False)


def _build_dual_adaptive_mask(
    raw: np.ndarray,
    threshold: float,
    cfg: PoreSegmentationConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    solid_mask = raw > threshold
    filled_solid = binary_fill_holes(solid_mask).astype(bool, copy=False)
    cavity_mask = filled_solid ^ solid_mask

    supplement_mask = np.zeros_like(cavity_mask, dtype=bool)
    supplement_enabled = False
    supplement_quantile_value: Optional[float] = None
    num_filled_components = 0

    if np.any(filled_solid):
        labels, num_filled_components = ndimage.label(filled_solid, structure=_STRUCTURE_26)
        if num_filled_components > 0:
            counts = np.bincount(labels.ravel())
            if counts.size > 1:
                counts[0] = 0
                largest_id = int(np.argmax(counts))
                largest_component = labels == largest_id
                inside_values = raw[largest_component]
                if inside_values.size > 0:
                    supplement_quantile_value = float(np.percentile(inside_values, cfg.supplement_quantile))
                    if supplement_quantile_value < float(threshold) - float(cfg.supplement_threshold_margin):
                        supplement_mask = largest_component & (raw <= supplement_quantile_value)
                        supplement_enabled = True

    pores_mask = cavity_mask | supplement_mask
    debug = {
        "num_filled_components": int(num_filled_components),
        "supplement_enabled": bool(supplement_enabled),
        "supplement_quantile_value": supplement_quantile_value,
        "supplement_voxels": int(np.count_nonzero(supplement_mask)),
        "cavity_voxels": int(np.count_nonzero(cavity_mask)),
    }
    return pores_mask.astype(bool, copy=False), debug


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(bool, copy=False)

    labels, num = ndimage.label(mask, structure=_STRUCTURE_26)
    if num == 0:
        return mask.astype(bool, copy=False)

    counts = np.bincount(labels.ravel())
    keep_ids = np.flatnonzero(counts >= int(min_size))
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    cleaned = np.isin(labels, keep_ids, assume_unique=False)
    return cleaned.astype(bool, copy=False)


def _build_adaptive_markers(
    distance_map: np.ndarray,
    pores_mask: np.ndarray,
    cfg: PoreSegmentationConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    comp_labels, num_components = ndimage.label(pores_mask, structure=_STRUCTURE_26)
    comp_slices = ndimage.find_objects(comp_labels)

    markers = np.zeros_like(comp_labels, dtype=np.int32)
    next_marker = 1
    num_forced = 0
    num_neck_split = 0

    for comp_id, slc in enumerate(comp_slices, start=1):
        if slc is None:
            continue

        local_component = comp_labels[slc] == comp_id
        voxel_count = int(np.count_nonzero(local_component))
        if voxel_count <= 0:
            continue

        local_dist = distance_map[slc]
        local_peaks = _component_peak_candidates(local_dist, local_component, voxel_count, cfg)
        peak_set = {tuple(int(v) for v in row) for row in local_peaks.tolist()}

        if not peak_set and cfg.force_seed_per_component:
            forced_peak = _argmax_coord(local_dist, local_component)
            if forced_peak is not None:
                peak_set.add(forced_peak)
                num_forced += 1

        if (
            cfg.neck_erosion_iters > 0
            and cfg.split_mode in {"balanced", "aggressive"}
            and voxel_count > cfg.small_component_voxels
        ):
            eroded = ndimage.binary_erosion(
                local_component,
                structure=_STRUCTURE_26,
                iterations=int(cfg.neck_erosion_iters),
                border_value=0,
            )
            core_labels, core_count = ndimage.label(eroded, structure=_STRUCTURE_26)
            if core_count > 1:
                for core_id in range(1, int(core_count) + 1):
                    core_mask = core_labels == core_id
                    if not np.any(core_mask):
                        continue
                    core_peak = _argmax_coord(local_dist, core_mask)
                    if core_peak is None:
                        continue
                    if core_peak not in peak_set:
                        peak_set.add(core_peak)
                        num_neck_split += 1

        if len(peak_set) > cfg.max_seeds_per_component:
            sorted_by_dist = sorted(peak_set, key=lambda p: float(local_dist[p]), reverse=True)
            peak_set = set(sorted_by_dist[: int(cfg.max_seeds_per_component)])

        for local_coord in sorted(peak_set, key=lambda p: float(local_dist[p]), reverse=True):
            gz = int(slc[0].start + local_coord[0])
            gy = int(slc[1].start + local_coord[1])
            gx = int(slc[2].start + local_coord[2])
            if markers[gz, gy, gx] != 0:
                continue
            markers[gz, gy, gx] = next_marker
            next_marker += 1

    if next_marker == 1:
        fallback_peak = _argmax_coord(distance_map, pores_mask)
        if fallback_peak is not None:
            markers[fallback_peak] = 1
            next_marker = 2
            num_forced += 1

    return markers, {
        "num_components": int(num_components),
        "num_forced_seeds": int(num_forced),
        "num_neck_split_seeds": int(num_neck_split),
        "num_markers": int(next_marker - 1),
    }


def _component_peak_candidates(
    local_dist: np.ndarray,
    local_component: np.ndarray,
    voxel_count: int,
    cfg: PoreSegmentationConfig,
) -> np.ndarray:
    min_distance = _component_min_distance(voxel_count, cfg)
    seed_cap = _component_seed_cap(voxel_count, cfg)
    peaks = peak_local_max(
        local_dist,
        min_distance=min_distance,
        labels=local_component,
        exclude_border=False,
    )
    if peaks.size == 0:
        return peaks

    values = local_dist[tuple(peaks.T)]
    component_max = float(np.max(local_dist[local_component])) if np.any(local_component) else 0.0
    keep_ratio = _PEAK_KEEP_RATIO.get(cfg.split_mode, _PEAK_KEEP_RATIO["balanced"])
    if component_max > 0.0:
        keep_mask = values >= (component_max * keep_ratio)
        if np.any(keep_mask):
            peaks = peaks[keep_mask]
            values = values[keep_mask]

    if len(peaks) > seed_cap:
        order = np.argsort(values)[::-1][: int(seed_cap)]
        peaks = peaks[order]
    return peaks


def _component_min_distance(voxel_count: int, cfg: PoreSegmentationConfig) -> int:
    if voxel_count <= cfg.small_component_voxels:
        return 1

    r_eq_vox = (3.0 * float(voxel_count) / (4.0 * np.pi)) ** (1.0 / 3.0)
    scale = _SPLIT_SCALE.get(cfg.split_mode, _SPLIT_SCALE["balanced"])
    min_dist = int(round(scale * r_eq_vox))
    if cfg.split_mode == "conservative":
        min_dist = max(min_dist, 2)
    return int(np.clip(min_dist, 1, int(cfg.base_min_peak_distance)))


def _component_seed_cap(voxel_count: int, cfg: PoreSegmentationConfig) -> int:
    if cfg.split_mode == "conservative":
        return 1

    if voxel_count <= cfg.small_component_voxels:
        return 1

    r_eq_vox = (3.0 * float(voxel_count) / (4.0 * np.pi)) ** (1.0 / 3.0)
    if cfg.split_mode == "aggressive":
        suggested = int(round(r_eq_vox / 2.2))
    else:
        suggested = int(round(r_eq_vox / 2.8))
    suggested = max(1, suggested)
    return int(min(int(cfg.max_seeds_per_component), suggested))


def _argmax_coord(dist_map: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    if not np.any(mask):
        return None
    masked = np.where(mask, dist_map, -np.inf)
    flat_idx = int(np.argmax(masked))
    if not np.isfinite(masked.ravel()[flat_idx]):
        return None
    return tuple(int(v) for v in np.unravel_index(flat_idx, masked.shape))


def _spacing_xyz_to_sampling_zyx(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(spacing_xyz) != 3:
        return (1.0, 1.0, 1.0)
    sx, sy, sz = spacing_xyz
    return float(sz), float(sy), float(sx)
