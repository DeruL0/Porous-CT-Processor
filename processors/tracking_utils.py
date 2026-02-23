"""
Utility helpers for robust 4D CT TGGA tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift as nd_shift
from core.gpu_backend import CUPY_AVAILABLE, get_gpu_backend

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage as gpu_ndimage
else:
    cp = None
    gpu_ndimage = None

try:
    from skimage.registration import phase_cross_correlation

    HAS_SKIMAGE_REGISTRATION = True
except Exception:
    phase_cross_correlation = None
    HAS_SKIMAGE_REGISTRATION = False


EPS = 1e-8


def bounded_volume_penalty(
    reference_volume: float,
    current_volume: float,
    mode: str = "symdiff",
    gaussian_sigma: float = 1.0,
) -> float:
    """
    Bounded volume mismatch penalty in [0, 1].

    Modes:
      - symdiff: |Vref - Vcur| / (Vref + Vcur)
      - gaussian: 1 - exp(-(Vref - Vcur)^2 / (2*sigma^2))
    """
    v_ref = max(float(reference_volume), 0.0)
    v_cur = max(float(current_volume), 0.0)
    resolved_mode = (mode or "symdiff").lower()

    if resolved_mode in {"gaussian", "inverse_gaussian", "anti_gaussian"}:
        sigma = max(float(gaussian_sigma), EPS)
        delta = v_ref - v_cur
        penalty = 1.0 - np.exp(-(delta * delta) / (2.0 * sigma * sigma))
        return float(np.clip(penalty, 0.0, 1.0))

    denom = max(v_ref + v_cur, EPS)
    penalty = abs(v_ref - v_cur) / denom
    return float(np.clip(penalty, 0.0, 1.0))


@dataclass
class MacroRegistrationResult:
    """Macro pre-registration output for gating and closure analysis."""

    displacement: np.ndarray
    compression_field: Optional[np.ndarray] = None
    method: str = "none"
    confidence: float = 0.0

    def warp_point(self, point_zyx: np.ndarray) -> np.ndarray:
        return np.asarray(point_zyx, dtype=np.float64) + np.asarray(self.displacement, dtype=np.float64)

    def sample_compression(self, point_zyx: np.ndarray, default: float = 0.0) -> float:
        if self.compression_field is None or self.compression_field.size == 0:
            return float(default)

        p = np.asarray(point_zyx, dtype=np.float64)
        if p.size != 3:
            return float(default)

        shape = np.asarray(self.compression_field.shape, dtype=np.float64) - 1.0
        clipped = np.clip(p, 0.0, np.maximum(shape, 0.0))
        value = map_coordinates(
            self.compression_field,
            [[clipped[0]], [clipped[1]], [clipped[2]]],
            order=1,
            mode="nearest",
        )
        return float(value[0]) if value.size else float(default)


def _phase_cross_correlation_gpu(ref_density_gpu, cur_density_gpu) -> Tuple[np.ndarray, float]:
    """
    Integer-shift phase correlation on GPU.

    Returns shift that maps current -> reference and a confidence score in [0, 1].
    """
    if not CUPY_AVAILABLE or cp is None:
        raise RuntimeError("CuPy is unavailable")

    eps = np.float32(1e-12)
    ref_fft = cp.fft.fftn(ref_density_gpu)
    cur_fft = cp.fft.fftn(cur_density_gpu)
    cross_power = ref_fft * cp.conj(cur_fft)
    cross_power /= cp.maximum(cp.abs(cross_power), eps)

    corr = cp.fft.ifftn(cross_power)
    corr_abs = cp.abs(corr)
    peak_flat = int(cp.argmax(corr_abs).get())
    peak_idx = np.unravel_index(peak_flat, corr_abs.shape)

    shift = np.asarray(peak_idx, dtype=np.float64)
    shape = np.asarray(corr_abs.shape, dtype=np.float64)
    half = shape / 2.0
    shift = np.where(shift > half, shift - shape, shift)

    peak_val = float(cp.asnumpy(corr_abs.ravel()[peak_flat]))
    mean_val = float(cp.asnumpy(cp.mean(corr_abs)))
    peak_ratio = peak_val / max(mean_val, EPS)
    confidence = float(np.clip((peak_ratio - 1.0) / 10.0, 0.0, 1.0))
    return shift, confidence


def estimate_macro_registration(
    reference_regions: Optional[np.ndarray],
    current_regions: Optional[np.ndarray],
    smoothing_sigma: float = 1.5,
    upsample_factor: int = 4,
    use_gpu: bool = False,
    gpu_min_size_mb: float = 8.0,
) -> MacroRegistrationResult:
    """
    Estimate macro displacement from previous frame to current frame.

    This is a DVC-style global registration using phase cross-correlation,
    with a center-of-mass fallback.
    """
    zero = np.zeros(3, dtype=np.float64)
    if reference_regions is None or current_regions is None:
        return MacroRegistrationResult(displacement=zero, method="missing_regions", confidence=0.0)
    if reference_regions.shape != current_regions.shape:
        return MacroRegistrationResult(displacement=zero, method="shape_mismatch", confidence=0.0)

    ref_binary = (reference_regions > 0).astype(np.float32)
    cur_binary = (current_regions > 0).astype(np.float32)

    com_displacement: Optional[np.ndarray] = None
    ref_coords = np.argwhere(ref_binary > 0)
    cur_coords = np.argwhere(cur_binary > 0)
    if len(ref_coords) > 0 and len(cur_coords) > 0:
        com_displacement = np.asarray(cur_coords.mean(axis=0) - ref_coords.mean(axis=0), dtype=np.float64)

    sigma = max(float(smoothing_sigma), 0.0)
    sanity_limit = (
        max(5.0, 0.25 * np.linalg.norm(com_displacement) + 2.0)
        if com_displacement is not None
        else 5.0
    )

    if use_gpu and CUPY_AVAILABLE and cp is not None and gpu_ndimage is not None:
        backend = get_gpu_backend()
        total_mb = (ref_binary.nbytes + cur_binary.nbytes) / (1024.0 * 1024.0)
        estimated_bytes = int((ref_binary.nbytes + cur_binary.nbytes) * 10)
        if backend.available and total_mb >= float(gpu_min_size_mb) and backend.can_fit(estimated_bytes, safety_factor=0.7):
            try:
                ref_binary_gpu = cp.asarray(ref_binary)
                cur_binary_gpu = cp.asarray(cur_binary)
                if sigma > 0.0:
                    ref_density_gpu = gpu_ndimage.gaussian_filter(ref_binary_gpu, sigma=sigma)
                    cur_density_gpu = gpu_ndimage.gaussian_filter(cur_binary_gpu, sigma=sigma)
                else:
                    ref_density_gpu = ref_binary_gpu
                    cur_density_gpu = cur_binary_gpu

                shift_to_reference_gpu, gpu_confidence = _phase_cross_correlation_gpu(
                    ref_density_gpu=ref_density_gpu,
                    cur_density_gpu=cur_density_gpu,
                )
                displacement = -np.asarray(shift_to_reference_gpu, dtype=np.float64)
                method = "phase_cross_correlation_gpu"
                confidence = float(gpu_confidence)

                if com_displacement is not None and np.linalg.norm(displacement - com_displacement) > sanity_limit:
                    displacement = com_displacement
                    method = "center_of_mass_fallback"
                    confidence = max(confidence, 0.35)

                warped_reference_density_gpu = gpu_ndimage.shift(
                    ref_density_gpu,
                    shift=tuple(float(v) for v in displacement),
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                compression_field_gpu = cp.clip(
                    (warped_reference_density_gpu - cur_density_gpu)
                    / cp.maximum(warped_reference_density_gpu, np.float32(EPS)),
                    0.0,
                    1.0,
                ).astype(cp.float32)
                compression_field = cp.asnumpy(compression_field_gpu)

                del ref_binary_gpu, cur_binary_gpu, ref_density_gpu, cur_density_gpu
                del warped_reference_density_gpu, compression_field_gpu
                backend.clear_memory(force=False)
                return MacroRegistrationResult(
                    displacement=np.asarray(displacement, dtype=np.float64),
                    compression_field=compression_field,
                    method=method,
                    confidence=confidence,
                )
            except Exception:
                backend.clear_memory(force=False)

    ref_density = gaussian_filter(ref_binary, sigma=sigma)
    cur_density = gaussian_filter(cur_binary, sigma=sigma)

    displacement = zero.copy()
    method = "zero"
    confidence = 0.0

    if HAS_SKIMAGE_REGISTRATION and np.any(ref_density) and np.any(cur_density):
        try:
            shift_to_reference, error, _ = phase_cross_correlation(
                ref_density,
                cur_density,
                upsample_factor=max(int(upsample_factor), 1),
            )
            # phase_cross_correlation returns current->reference shift.
            displacement = -np.asarray(shift_to_reference, dtype=np.float64)
            method = "phase_cross_correlation"
            confidence = float(np.clip(1.0 - float(error), 0.0, 1.0))
        except Exception:
            displacement = zero.copy()

    if method == "phase_cross_correlation" and com_displacement is not None:
        if np.linalg.norm(displacement - com_displacement) > sanity_limit:
            displacement = com_displacement
            method = "center_of_mass_fallback"
            confidence = max(confidence, 0.35)

    if method == "zero":
        if com_displacement is not None:
            displacement = com_displacement
            method = "center_of_mass"
            confidence = 0.35
        elif len(ref_coords) > 0 and len(cur_coords) == 0:
            method = "empty_current"
            confidence = 0.0
        else:
            method = "empty_reference"
            confidence = 0.0

    warped_reference_density = nd_shift(
        ref_density,
        shift=displacement,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    compression_field = np.clip(
        (warped_reference_density - cur_density) / np.maximum(warped_reference_density, EPS),
        0.0,
        1.0,
    ).astype(np.float32)

    return MacroRegistrationResult(
        displacement=np.asarray(displacement, dtype=np.float64),
        compression_field=compression_field,
        method=method,
        confidence=confidence,
    )


def extract_shifted_overlap_region(
    current_regions: np.ndarray,
    local_reference_mask: np.ndarray,
    bbox_mins: np.ndarray,
    shift_zyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract aligned local region by shifting the reference bbox in voxel-space.

    Returns:
      (shifted_ref_mask, shifted_current_region) with identical shape.
    """
    shape = np.asarray(local_reference_mask.shape, dtype=np.int64)
    if shape.size != 3 or np.any(shape <= 0):
        return np.zeros((0, 0, 0), dtype=bool), np.zeros((0, 0, 0), dtype=current_regions.dtype)

    mins = np.asarray(bbox_mins, dtype=np.int64)
    shift_zyx = np.asarray(np.rint(shift_zyx), dtype=np.int64)

    target_mins = mins + shift_zyx
    target_maxs = target_mins + shape

    image_shape = np.asarray(current_regions.shape, dtype=np.int64)
    clipped_mins = np.maximum(target_mins, 0)
    clipped_maxs = np.minimum(target_maxs, image_shape)
    clipped_shape = clipped_maxs - clipped_mins
    if np.any(clipped_shape <= 0):
        return np.zeros((0, 0, 0), dtype=bool), np.zeros((0, 0, 0), dtype=current_regions.dtype)

    mask_offset_min = clipped_mins - target_mins
    mask_offset_max = mask_offset_min + clipped_shape

    ref_local = local_reference_mask[
        mask_offset_min[0]:mask_offset_max[0],
        mask_offset_min[1]:mask_offset_max[1],
        mask_offset_min[2]:mask_offset_max[2],
    ]
    curr_local = current_regions[
        clipped_mins[0]:clipped_maxs[0],
        clipped_mins[1]:clipped_maxs[1],
        clipped_mins[2]:clipped_maxs[2],
    ]
    return ref_local, curr_local


def should_mark_closed_by_compression(
    previous_volume: float,
    reference_volume: float,
    local_compression: float,
    volume_ratio_threshold: float = 0.02,
    min_volume_voxels: float = 2.0,
    compression_threshold: float = 0.6,
) -> bool:
    """Rule-based closure decision for a physically disappearing pore."""
    ref_volume_safe = max(float(reference_volume), EPS)
    prev_volume_safe = max(float(previous_volume), 0.0)
    ratio = prev_volume_safe / ref_volume_safe

    tiny_last_frame = (
        prev_volume_safe <= float(min_volume_voxels)
        or ratio <= max(float(volume_ratio_threshold), 0.0)
    )
    high_compression_zone = float(local_compression) >= float(compression_threshold)
    return bool(tiny_last_frame and high_compression_zone)


class ConstantAccelerationKalman3D:
    """Constant-acceleration Kalman filter with state [p, v, a] in 3D."""

    def __init__(
        self,
        initial_position: np.ndarray,
        dt: float = 1.0,
        process_noise: float = 0.05,
        measurement_noise: float = 1.0,
    ) -> None:
        self.dt = float(max(dt, 1e-6))
        self.process_noise = float(max(process_noise, 1e-9))
        self.measurement_noise = float(max(measurement_noise, 1e-9))

        self.state = np.zeros((9, 1), dtype=np.float64)
        self.state[0:3, 0] = np.asarray(initial_position, dtype=np.float64)

        self.covariance = np.eye(9, dtype=np.float64)
        self.covariance[3:6, 3:6] *= 5.0
        self.covariance[6:9, 6:9] *= 10.0

        self.measurement_matrix = np.zeros((3, 9), dtype=np.float64)
        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_matrix[2, 2] = 1.0
        self.measurement_covariance = np.eye(3, dtype=np.float64) * self.measurement_noise

        self.transition_matrix = self._build_transition(self.dt)
        self.process_covariance = self._build_process_covariance(self.dt, self.process_noise)

    def _build_transition(self, dt: float) -> np.ndarray:
        f = np.eye(9, dtype=np.float64)
        for axis in range(3):
            pos = axis
            vel = axis + 3
            acc = axis + 6
            f[pos, vel] = dt
            f[pos, acc] = 0.5 * dt * dt
            f[vel, acc] = dt
        return f

    def _build_process_covariance(self, dt: float, noise: float) -> np.ndarray:
        q = np.zeros((9, 9), dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        base = np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=np.float64,
        ) * noise

        for axis in range(3):
            ids = [axis, axis + 3, axis + 6]
            for r in range(3):
                for c in range(3):
                    q[ids[r], ids[c]] = base[r, c]
        return q

    def _refresh_dynamics(self, dt: float) -> None:
        self.dt = float(max(dt, 1e-6))
        self.transition_matrix = self._build_transition(self.dt)
        self.process_covariance = self._build_process_covariance(self.dt, self.process_noise)

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        if dt is not None:
            self._refresh_dynamics(float(dt))

        self.state = self.transition_matrix @ self.state
        self.covariance = (
            self.transition_matrix @ self.covariance @ self.transition_matrix.T
            + self.process_covariance
        )
        return self.current_position()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=np.float64).reshape(3, 1)
        innovation = z - self.measurement_matrix @ self.state
        innovation_cov = (
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T
            + self.measurement_covariance
        )
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation

        identity = np.eye(9, dtype=np.float64)
        residual = identity - kalman_gain @ self.measurement_matrix
        # Joseph form for numerical stability.
        self.covariance = (
            residual @ self.covariance @ residual.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )
        return self.current_position()

    def current_position(self) -> np.ndarray:
        return self.state[0:3, 0].copy()

    def apply_brake(
        self,
        miss_count: int,
        velocity_decay: float = 0.75,
        acceleration_decay: float = 0.35,
        freeze_after_misses: int = 3,
    ) -> np.ndarray:
        """
        Brake motion when detections are missing to prevent runaway drift.

        The filter is still allowed to coast for short occlusions, but velocity
        and acceleration are exponentially damped and fully frozen after a
        configurable number of consecutive misses.
        """
        misses = max(int(miss_count), 1)
        v_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        a_decay = float(np.clip(acceleration_decay, 0.0, 1.0))
        freeze_after = max(int(freeze_after_misses), 1)

        if misses >= freeze_after:
            self.state[3:6, 0] = 0.0
            self.state[6:9, 0] = 0.0
        else:
            self.state[3:6, 0] *= v_decay
            self.state[6:9, 0] *= a_decay
        return self.current_position()

    def set_position(self, position: np.ndarray) -> None:
        self.state[0:3, 0] = np.asarray(position, dtype=np.float64)
