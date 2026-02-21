"""
HPC-grade 3-D spatial chunker with ghost-cell (halo) support.

Design goals
------------
* Eliminate TLB-thrashing caused by operating on large numpy arrays that do
  not fit in LLC (Last-Level Cache) or physical RAM.
* Provide correct boundary handling for non-local algorithms (EDT, convolution,
  morphological operations) via configurable halo/ghost-cell overlap.
* Expose a simple DAG-ready generator interface that yields ``ChunkDescriptor``
  objects; callers compose processing stages as pure functions.

No PyQt5, no VTK, no CuPy imports — this module is completely headless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable, Generator, Iterable, Optional, Sequence, Tuple
)

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkDescriptor:
    """
    Describes a single spatial chunk and the surrounding halo region.

    Attributes
    ----------
    chunk_id : int
        Monotonically increasing identifier (useful for progress reporting).
    volume_shape : tuple(int, int, int)
        Full volume shape (Z, Y, X).
    core_slices : tuple of 3 slices
        Slices that select the *valid* (non-halo) output region within the
        extended (with-halo) array.
    extended_slices : tuple of 3 slices
        Slices that select the with-halo input region from the full volume.
    core_in_extended : tuple of 3 slices
        Slices that locate the core region *within* the extended array.
        Use these to strip the halo after processing.
    halo : int
        Halo width in voxels (same on all sides).
    """

    chunk_id:          int
    volume_shape:      Tuple[int, int, int]
    core_slices:       Tuple[slice, slice, slice]
    extended_slices:   Tuple[slice, slice, slice]
    core_in_extended:  Tuple[slice, slice, slice]
    halo:              int

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cs = self.core_slices
        return (
            f"ChunkDescriptor(id={self.chunk_id}, "
            f"core=[{cs[0].start}:{cs[0].stop}, "
            f"{cs[1].start}:{cs[1].stop}, "
            f"{cs[2].start}:{cs[2].stop}], "
            f"halo={self.halo})"
        )

    @property
    def core_shape(self) -> Tuple[int, int, int]:
        return tuple(s.stop - s.start for s in self.core_slices)   # type: ignore[return-value]

    @property
    def extended_shape(self) -> Tuple[int, int, int]:
        return tuple(s.stop - s.start for s in self.extended_slices)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SpatialChunker
# ---------------------------------------------------------------------------

class SpatialChunker:
    """
    Enumerate 3-D chunks with optional ghost-cell borders.

    Parameters
    ----------
    volume_shape : (D, H, W)
        Shape of the full volume.
    chunk_shape : (cd, ch, cw)
        Desired shape of each core chunk.  The final chunk in each dimension
        may be smaller.
    halo : int
        Number of ghost-cell voxels to add on every face of every chunk.
        Algorithms reading ``halo`` voxels beyond their core boundary will
        always receive valid data (edge voxels are clamped to the volume
        boundary — no zero-padding artefacts).
    """

    def __init__(
        self,
        volume_shape:  Tuple[int, int, int],
        chunk_shape:   Tuple[int, int, int] = (128, 128, 128),
        halo:          int                  = 0,
    ) -> None:
        assert len(volume_shape) == 3,  "volume_shape must be 3-D"
        assert len(chunk_shape)  == 3,  "chunk_shape must be 3-D"
        assert halo >= 0,               "halo must be non-negative"

        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.chunk_shape  = tuple(int(c) for c in chunk_shape)
        self.halo         = int(halo)

        D, H, W   = self.volume_shape
        cd, ch, cw = self.chunk_shape

        self._starts_z = list(range(0, D, cd))
        self._starts_y = list(range(0, H, ch))
        self._starts_x = list(range(0, W, cw))

    # ------------------------------------------------------------------
    @property
    def num_chunks(self) -> int:
        return len(self._starts_z) * len(self._starts_y) * len(self._starts_x)

    # ------------------------------------------------------------------
    def __iter__(self) -> Generator[ChunkDescriptor, None, None]:
        """Yield ChunkDescriptor objects for every chunk in ZYX order."""
        D, H, W    = self.volume_shape
        cd, ch, cw = self.chunk_shape
        h          = self.halo
        chunk_id   = 0

        for z0 in self._starts_z:
            z1 = min(z0 + cd, D)
            for y0 in self._starts_y:
                y1 = min(y0 + ch, H)
                for x0 in self._starts_x:
                    x1 = min(x0 + cw, W)

                    # Core region in full-volume coordinates
                    core_slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

                    # Extended region (with halo, clamped to array bounds)
                    ez0 = max(z0 - h, 0);  ez1 = min(z1 + h, D)
                    ey0 = max(y0 - h, 0);  ey1 = min(y1 + h, H)
                    ex0 = max(x0 - h, 0);  ex1 = min(x1 + h, W)
                    extended_slices = (slice(ez0, ez1), slice(ey0, ey1), slice(ex0, ex1))

                    # Where is the core inside the extended array?
                    cz0_in = z0 - ez0;  cz1_in = cz0_in + (z1 - z0)
                    cy0_in = y0 - ey0;  cy1_in = cy0_in + (y1 - y0)
                    cx0_in = x0 - ex0;  cx1_in = cx0_in + (x1 - x0)
                    core_in_extended = (
                        slice(cz0_in, cz1_in),
                        slice(cy0_in, cy1_in),
                        slice(cx0_in, cx1_in),
                    )

                    yield ChunkDescriptor(
                        chunk_id         = chunk_id,
                        volume_shape     = self.volume_shape,
                        core_slices      = core_slices,
                        extended_slices  = extended_slices,
                        core_in_extended = core_in_extended,
                        halo             = h,
                    )
                    chunk_id += 1

    # ------------------------------------------------------------------
    def process_inplace(
        self,
        volume:   np.ndarray,
        fn:       Callable[[np.ndarray, ChunkDescriptor], np.ndarray],
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """
        Apply *fn* to every chunk in-place, stripping halos before write-back.

        Parameters
        ----------
        volume : ndarray, shape == self.volume_shape
            The array to modify in-place.
        fn : callable(extended_chunk, desc) -> processed_chunk
            Must accept the with-halo sub-array and the descriptor, and return
            an array shaped like the *extended* input.  The core region will
            be extracted automatically using ``desc.core_in_extended``.
        progress : optional callable(percent: int, message: str)
            Called after each chunk with a completion percentage.
        """
        total = self.num_chunks
        for desc in self:
            ext_chunk = volume[desc.extended_slices].copy()
            result    = fn(ext_chunk, desc)
            # Write back only the core (halo stripped)
            volume[desc.core_slices] = result[desc.core_in_extended]

            if progress:
                pct = int(100 * (desc.chunk_id + 1) / total)
                progress(pct, f"Chunk {desc.chunk_id + 1}/{total}")

    # ------------------------------------------------------------------
    def map_reduce(
        self,
        volume:   np.ndarray,
        map_fn:   Callable[[np.ndarray, ChunkDescriptor], Any],
        reduce_fn: Callable[[Iterable], Any],
        progress: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Map *map_fn* over chunks and reduce with *reduce_fn*.

        Useful for aggregation tasks (histogram, statistics) without
        loading the entire volume at once.
        """
        import builtins
        from typing import Any

        total    = self.num_chunks
        partials = []
        for desc in self:
            ext_chunk = volume[desc.extended_slices]
            partials.append(map_fn(ext_chunk, desc))
            if progress:
                pct = int(100 * (desc.chunk_id + 1) / total)
                progress(pct, f"Chunk {desc.chunk_id + 1}/{total}")
        return reduce_fn(partials)


# ---------------------------------------------------------------------------
# Convenience: EDT with ghost cells
# ---------------------------------------------------------------------------

def edt_chunked(
    binary_mask:   np.ndarray,
    chunk_shape:   Tuple[int, int, int] = (128, 128, 128),
    halo:          int                  = 16,
    sampling:      Optional[Tuple[float, ...]] = None,
    progress:      Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    """
    Memory-efficient 3-D Euclidean Distance Transform via ghost-cell chunking.

    A halo of ``halo`` voxels on every face ensures that EDT values computed
    at chunk boundaries are accurate to within ``halo`` voxels of the true
    global EDT.  For most medical-imaging scenarios halo=16 is sufficient.

    Parameters
    ----------
    binary_mask : bool / uint8 ndarray
        Input mask (True = foreground).
    chunk_shape : (D, H, W)
        Core chunk size.  Larger = faster (less overhead), more RAM.
    halo : int
        Ghost-cell width.  Should be ≥ max expected pore radius.
    sampling : optional tuple of floats
        Voxel spacing (z, y, x) for anisotropic EDT.
    progress : optional progress callback

    Returns
    -------
    dist : float32 ndarray, same shape as binary_mask
    """
    import scipy.ndimage as ndi

    dist = np.zeros(binary_mask.shape, dtype=np.float32)
    chunker = SpatialChunker(binary_mask.shape, chunk_shape=chunk_shape, halo=halo)  # type: ignore[arg-type]

    def _edt_chunk(ext_chunk: np.ndarray, desc: ChunkDescriptor) -> np.ndarray:
        result = ndi.distance_transform_edt(ext_chunk, sampling=sampling).astype(np.float32)
        return result

    chunker.process_inplace(
        volume   = dist,
        fn       = lambda _, desc: _run_edt_on_mask(binary_mask, desc, sampling),
        progress = progress,
    )
    return dist


def _run_edt_on_mask(mask, desc: ChunkDescriptor, sampling) -> np.ndarray:
    """Internal helper used by edt_chunked."""
    import scipy.ndimage as ndi
    ext_chunk = mask[desc.extended_slices].copy()
    return ndi.distance_transform_edt(ext_chunk, sampling=sampling).astype(np.float32)


# ---------------------------------------------------------------------------
# Lightweight DAG executor  (no external dependencies)
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A single step in a processing pipeline."""
    name:        str
    fn:          Callable[[Any], Any]
    depends_on:  Tuple[str, ...] = field(default_factory=tuple)


class SimpleDAGExecutor:
    """
    Topologically-sorted pipeline runner.

    Usage::

        dag = SimpleDAGExecutor()
        dag.add(DAGNode("load",     load_fn,    depends_on=()))
        dag.add(DAGNode("segment",  segment_fn, depends_on=("load",)))
        dag.add(DAGNode("pnm",      pnm_fn,     depends_on=("segment",)))
        results = dag.run(progress_callback)

    Each ``fn`` receives a dict of ``{node_name: result}`` for all nodes it
    depends on.  The return value is stored in the results dict under its own
    name and forwarded to dependent nodes.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add(self, node: DAGNode) -> "SimpleDAGExecutor":
        self._nodes[node.name] = node
        return self

    def _topo_sort(self) -> list[str]:
        visited:    set[str]  = set()
        order:      list[str] = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._nodes[name].depends_on:
                if dep not in self._nodes:
                    raise KeyError(f"DAG node '{name}' depends on unknown node '{dep}'")
                dfs(dep)
            order.append(name)

        for name in self._nodes:
            dfs(name)
        return order

    def run(
        self,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        order    = self._topo_sort()
        results  = {}
        total    = len(order)
        for i, name in enumerate(order):
            node    = self._nodes[name]
            inputs  = {dep: results[dep] for dep in node.depends_on}
            if progress:
                progress(int(100 * i / total), f"Running: {name}")
            results[name] = node.fn(inputs)
        if progress:
            progress(100, "Pipeline complete")
        return results
