"""
Regression & performance tests for the optimised GPU pipeline.

Covers
------
* watershed_kernel  — shared-memory / hierarchical-reduction version
* _nms_gpu_parallel — spatial-hash O(N) NMS

Run
---
    cd "d:/Projects/6.Food CT/Porous"
    python -m pytest tests/test_gpu_backend.py -v

Or standalone:
    python tests/test_gpu_backend.py
"""

import sys
import time
import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Minimal bootstrap so tests run from any cwd
# ---------------------------------------------------------------------------
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# GPU availability guard
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ===========================================================================
# Helpers
# ===========================================================================

def _make_two_blob_volume(shape=(64, 64, 64)):
    """Create a synthetic 3-D volume with two spherical blobs."""
    d, h, w = shape
    z, y, x = np.ogrid[:d, :h, :w]
    c1 = (d // 4,     h // 2, w // 2)
    c2 = (3 * d // 4, h // 2, w // 2)
    r  = min(d, h, w) // 5
    mask = (
        ((z - c1[0])**2 + (y - c1[1])**2 + (x - c1[2])**2 < r**2) |
        ((z - c2[0])**2 + (y - c2[1])**2 + (x - c2[2])**2 < r**2)
    )
    image = np.zeros(shape, dtype=np.float32)
    image[mask] = 1.0
    dist   = ndimage.distance_transform_edt(image).astype(np.float32)
    markers = np.zeros(shape, dtype=np.int32)
    markers[c1] = 1
    markers[c2] = 2
    return -dist, markers, mask.astype(bool)


def _make_random_peaks(n=200, shape=(64, 64, 64), min_sep=5, seed=42):
    """Create N random 3-D points with random float intensities."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, [s - 1 for s in shape], size=(n, 3))
    intensities = rng.random(n).astype(np.float32)
    return pts, intensities


def _kdtree_nms_reference(pts, intensities, min_distance):
    """Pure-CPU NMS via cKDTree — used as ground truth."""
    n = len(pts)
    order     = np.argsort(-intensities)
    pts_s     = pts[order]
    intens_s  = intensities[order]
    tree      = cKDTree(pts_s)
    suppressed = np.zeros(n, dtype=bool)
    selected   = []
    for i in range(n):
        if suppressed[i]:
            continue
        selected.append(pts_s[i])
        for j in tree.query_ball_point(pts_s[i], r=min_distance, p=2):
            if j > i:
                suppressed[j] = True
    return np.array(selected) if selected else np.empty((0, 3), dtype=pts.dtype)


# ===========================================================================
# Assertion 1 — Watershed label correctness
# ===========================================================================

class TestWatershedCorrectness:
    """GPU watershed must reproduce CPU watershed labels (>99 % agreement)."""

    def test_small_volume_label_match(self):
        from processors.utils import watershed_gpu
        from skimage.segmentation import watershed as cpu_watershed

        image, markers, mask = _make_two_blob_volume(shape=(48, 48, 48))

        labels_cpu = cpu_watershed(image, markers, mask=mask)
        labels_gpu = watershed_gpu(image, markers, mask=mask)

        total   = int(mask.sum())
        match   = int(np.sum((labels_cpu == labels_gpu) & mask))
        pct     = match / total if total > 0 else 1.0
        print(f"\n  Watershed label match: {pct*100:.2f}% ({match}/{total})")

        assert pct >= 0.99, (
            f"GPU watershed accuracy {pct*100:.2f}% < 99 % — "
            "shared-memory kernel may have off-by-one or indexing bug"
        )

    def test_label_values_bounded(self):
        """No label value should exceed the number of seeds."""
        from processors.utils import watershed_gpu

        image, markers, mask = _make_two_blob_volume(shape=(32, 32, 32))
        labels = watershed_gpu(image, markers, mask=mask)
        n_seeds = int(markers.max())
        assert int(labels.max()) <= n_seeds, "Unexpected label > n_seeds"
        assert int(labels.min()) >= 0, "Negative label in output"

    @staticmethod
    def run():
        t = TestWatershedCorrectness()
        t.test_small_volume_label_match()
        t.test_label_values_bounded()
        print("  [PASS] Watershed correctness")


# ===========================================================================
# Assertion 2 — NMS output equality with cKDTree reference
# ===========================================================================

class TestNmsCorrectness:
    """_nms_gpu_parallel peaks must match cKDTree NMS (set equality)."""

    def _pts_to_set(self, pts):
        return {tuple(p) for p in pts.tolist()}

    def test_nms_matches_kdtree(self):
        if not GPU_AVAILABLE:
            print("  [SKIP] CuPy not available — skipping GPU NMS test")
            return

        from processors.utils import _nms_gpu_parallel

        pts, intensities = _make_random_peaks(n=300, min_sep=5)
        min_dist = 5

        # ---- GPU result ----
        pts_gpu    = cp.asarray(pts.astype(np.int32))
        intens_gpu = cp.asarray(intensities)
        sort_idx   = cp.argsort(-intens_gpu)
        keep_gpu   = _nms_gpu_parallel(pts_gpu[sort_idx], intens_gpu[sort_idx], min_dist)
        sel_gpu    = cp.asnumpy(pts_gpu[sort_idx][keep_gpu])

        # ---- CPU reference ----
        sel_cpu    = _kdtree_nms_reference(pts, intensities, min_dist)

        gpu_set = self._pts_to_set(sel_gpu)
        cpu_set = self._pts_to_set(sel_cpu)

        # Points in CPU result but not GPU (false negatives)
        missed    = cpu_set - gpu_set
        # Points in GPU result but not CPU (false positives — should not happen)
        extra     = gpu_set - cpu_set

        print(
            f"\n  NMS — CPU kept: {len(cpu_set)}, GPU kept: {len(gpu_set)}, "
            f"missed: {len(missed)}, extra: {len(extra)}"
        )

        assert len(extra)  == 0, f"GPU NMS kept {len(extra)} extra points not in CPU reference"
        assert len(missed) == 0, f"GPU NMS missed {len(missed)} points present in CPU reference"

    def test_nms_empty_input(self):
        if not GPU_AVAILABLE:
            return
        from processors.utils import _nms_gpu_parallel
        empty_pts    = cp.empty((0, 3), dtype=cp.int32)
        empty_intens = cp.empty((0,),   dtype=cp.float32)
        keep         = _nms_gpu_parallel(empty_pts, empty_intens, min_distance=5)
        assert keep.shape == (0,), "Empty input should return empty keep array"

    def test_nms_single_point(self):
        if not GPU_AVAILABLE:
            return
        from processors.utils import _nms_gpu_parallel
        pts    = cp.array([[10, 20, 30]], dtype=cp.int32)
        intens = cp.array([0.9],         dtype=cp.float32)
        keep   = _nms_gpu_parallel(pts, intens, min_distance=5)
        assert bool(keep[0]) is True, "Single point must always be kept"

    @staticmethod
    def run():
        t = TestNmsCorrectness()
        t.test_nms_matches_kdtree()
        t.test_nms_empty_input()
        t.test_nms_single_point()
        print("  [PASS] NMS correctness")


# ===========================================================================
# Performance benchmark
# ===========================================================================

class BenchmarkGPUPipeline:
    """
    Throughput comparison on larger volumes.
    Prints timing; does NOT assert speed (hardware-dependent).
    """

    def benchmark_watershed(self, shape=(128, 128, 128)):
        from processors.utils import watershed_gpu
        from skimage.segmentation import watershed as cpu_watershed

        image, markers, mask = _make_two_blob_volume(shape=shape)

        t0 = time.perf_counter()
        cpu_watershed(image, markers, mask=mask)
        cpu_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        watershed_gpu(image, markers, mask=mask)
        gpu_time = time.perf_counter() - t0

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(
            f"\n  Watershed {shape}  "
            f"CPU {cpu_time:.3f}s | GPU {gpu_time:.3f}s | "
            f"speedup {speedup:.1f}x"
        )
        return speedup

    def benchmark_nms(self, n_points=50_000, min_dist=5):
        if not GPU_AVAILABLE:
            print("  [SKIP] CuPy not available — skipping GPU NMS benchmark")
            return

        from processors.utils import _nms_gpu_parallel

        rng       = np.random.default_rng(0)
        pts       = rng.integers(0, 512, size=(n_points, 3)).astype(np.int32)
        intensities = rng.random(n_points).astype(np.float32)

        # CPU KDTree NMS
        t0 = time.perf_counter()
        _kdtree_nms_reference(pts, intensities, min_dist)
        cpu_time = time.perf_counter() - t0

        # GPU spatial-hash NMS
        pts_gpu    = cp.asarray(pts)
        intens_gpu = cp.asarray(intensities)
        sort_idx   = cp.argsort(-intens_gpu)
        cp.cuda.Stream.null.synchronize()  # warm up

        t0 = time.perf_counter()
        keep = _nms_gpu_parallel(pts_gpu[sort_idx], intens_gpu[sort_idx], min_dist)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - t0

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(
            f"\n  NMS N={n_points} min_dist={min_dist}  "
            f"CPU {cpu_time:.3f}s | GPU {gpu_time:.3f}s | "
            f"speedup {speedup:.1f}x  "
            f"kept {int(keep.sum())}/{n_points}"
        )

    @staticmethod
    def run():
        b = BenchmarkGPUPipeline()
        b.benchmark_watershed(shape=(64,  64,  64))
        b.benchmark_watershed(shape=(128, 128, 128))
        b.benchmark_nms(n_points=10_000)
        b.benchmark_nms(n_points=50_000)
        print("  [PASS] Benchmark complete (see timings above)")


# ===========================================================================
# Pytest entry points (picked up by pytest auto-discovery)
# ===========================================================================

def test_watershed_label_match():
    TestWatershedCorrectness().test_small_volume_label_match()

def test_watershed_label_bounds():
    TestWatershedCorrectness().test_label_values_bounded()

def test_nms_matches_kdtree():
    TestNmsCorrectness().test_nms_matches_kdtree()

def test_nms_empty_input():
    TestNmsCorrectness().test_nms_empty_input()

def test_nms_single_point():
    TestNmsCorrectness().test_nms_single_point()


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPU Backend Regression & Benchmark Suite")
    print(f"CuPy available: {GPU_AVAILABLE}")
    print("=" * 60)

    print("\n--- Assertion 1: Watershed correctness ---")
    TestWatershedCorrectness.run()

    print("\n--- Assertion 2: NMS correctness ---")
    TestNmsCorrectness.run()

    print("\n--- Performance benchmarks ---")
    BenchmarkGPUPipeline.run()

    print("\n" + "=" * 60)
    print("All checks passed.")
