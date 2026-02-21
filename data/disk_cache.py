"""
Unified disk cache subsystem for large volume and time-series data.

This module centralizes filesystem-backed cache behavior behind reusable
storage backends, then composes domain caches on top:
  - DiskCacheManager      -> memmap backend
  - SegmentationCache     -> memmap backend
  - TimeSeriesPNMCache    -> pickle backend
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_key(raw_key: str, max_prefix_len: int = 80) -> str:
    """
    Convert any key string into a filesystem-safe token.
    """
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(raw_key))
    if len(cleaned) > max_prefix_len:
        cleaned = cleaned[:max_prefix_len]
    digest = hashlib.md5(str(raw_key).encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{digest}"


def _remove_file(path: str) -> None:
    """Best-effort file removal with logging."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Failed to remove cache file %s: %s", path, exc)


class StorageBackend(ABC):
    """
    Abstract filesystem backend used by concrete cache managers.
    """

    def __init__(self, cache_dir: Optional[str], prefix: str, extension: str) -> None:
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.prefix = prefix
        self.extension = extension
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_path(self, key: str) -> str:
        safe = _safe_key(key)
        return os.path.join(self.cache_dir, f"{self.prefix}_{safe}.{self.extension}")

    def exists(self, entry: Dict[str, Any]) -> bool:
        return os.path.exists(entry["path"])

    @abstractmethod
    def write(self, key: str, value: Any) -> Dict[str, Any]:
        """Persist a value and return backend metadata entry."""

    @abstractmethod
    def read(self, entry: Dict[str, Any]) -> Any:
        """Load value from backend entry."""

    def delete(self, entry: Dict[str, Any]) -> None:
        _remove_file(entry["path"])


class MemmapStorageBackend(StorageBackend):
    """
    Storage backend for numpy arrays via `np.memmap` files.
    """

    def __init__(self, cache_dir: Optional[str] = None, prefix: str = "porous_memmap") -> None:
        super().__init__(cache_dir=cache_dir, prefix=prefix, extension="dat")

    def create_empty(self, key: str, shape: Tuple[int, ...], dtype=np.float32) -> Tuple[np.memmap, Dict[str, Any]]:
        path = self.build_path(key)
        mmap = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        entry = {
            "path": path,
            "shape": tuple(shape),
            "dtype": np.dtype(dtype).str,
            "kind": "memmap",
        }
        return mmap, entry

    def write(self, key: str, value: np.ndarray) -> Dict[str, Any]:
        mmap, entry = self.create_empty(key=key, shape=value.shape, dtype=value.dtype)
        mmap[:] = value[:]
        mmap.flush()
        return entry

    def read(self, entry: Dict[str, Any]) -> Optional[np.memmap]:
        path = entry["path"]
        if not os.path.exists(path):
            return None
        return np.memmap(
            path,
            dtype=np.dtype(entry["dtype"]),
            mode="r",
            shape=tuple(entry["shape"]),
        )


class PickleStorageBackend(StorageBackend):
    """
    Storage backend for arbitrary Python objects via pickle files.
    """

    def __init__(self, cache_dir: Optional[str] = None, prefix: str = "porous_pickle") -> None:
        super().__init__(cache_dir=cache_dir, prefix=prefix, extension="pkl")

    def write(self, key: str, value: Any) -> Dict[str, Any]:
        path = self.build_path(key)
        with open(path, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return {"path": path, "kind": "pickle"}

    def read(self, entry: Dict[str, Any]) -> Any:
        path = entry["path"]
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)


class CacheRegistry:
    """
    Centralized singleton registry for runtime cache instances.
    """

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}

    def get(self, name: str, factory):
        if name not in self._instances:
            self._instances[name] = factory()
        return self._instances[name]

    def clear(self, name: str, *, drop_instance: bool = False) -> None:
        instance = self._instances.get(name)
        if instance is None:
            return
        try:
            if hasattr(instance, "release_all"):
                instance.release_all()
            elif hasattr(instance, "clear"):
                instance.clear()
        finally:
            if drop_instance:
                self._instances.pop(name, None)
            gc.collect()

    def clear_all(self, *, drop_instances: bool = False) -> None:
        for name in list(self._instances.keys()):
            self.clear(name, drop_instance=drop_instances)


class DiskCacheManager:
    """
    General-purpose memmap cache manager for large numpy arrays.
    """

    def __init__(self, cache_dir: Optional[str] = None, prefix: str = "porous_cache") -> None:
        self.backend = MemmapStorageBackend(cache_dir=cache_dir, prefix=prefix)
        self._cache_entries: Dict[str, Dict[str, Any]] = {}
        self._cache_arrays: Dict[str, np.memmap] = {}
        self._cache_files: Dict[str, str] = {}

    def create_array(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> np.memmap:
        self.release(name)
        mmap, entry = self.backend.create_empty(name, shape=shape, dtype=dtype)
        self._cache_entries[name] = entry
        self._cache_arrays[name] = mmap
        self._cache_files[name] = entry["path"]
        print(f"[DiskCache] Created '{name}': {shape}, {dtype}, {self._format_size(mmap.nbytes)}")
        return mmap

    def get_array(self, name: str) -> Optional[np.memmap]:
        arr = self._cache_arrays.get(name)
        if arr is not None:
            return arr
        entry = self._cache_entries.get(name)
        if entry is None:
            return None
        reopened = self.backend.read(entry)
        if reopened is not None:
            self._cache_arrays[name] = reopened
        return reopened

    def release(self, name: str) -> None:
        if name in self._cache_arrays:
            del self._cache_arrays[name]
        entry = self._cache_entries.pop(name, None)
        if entry is not None:
            self.backend.delete(entry)
        self._cache_files.pop(name, None)
        gc.collect()

    def release_all(self) -> None:
        for name in list(self._cache_entries.keys()):
            self.release(name)
        gc.collect()

    def _format_size(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def __del__(self) -> None:
        try:
            self.release_all()
        except Exception as exc:
            logger.debug("DiskCacheManager cleanup during destruction failed: %s", exc)


class ChunkedProcessor:
    """
    Process volumes in chunks and persist output via disk-backed memmap.
    """

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
        self.cache = DiskCacheManager()

    def process_chunked(
        self,
        volume: np.ndarray,
        func,
        output_dtype=np.float32,
        callback=None,
    ) -> np.memmap:
        shape = volume.shape
        n_slices = shape[0]
        n_chunks = (n_slices + self.chunk_size - 1) // self.chunk_size
        output = self.cache.create_array("output", shape, dtype=output_dtype)

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, n_slices)
            chunk = volume[start:end]
            result = func(chunk)
            output[start:end] = result
            output.flush()

            if callback:
                progress = int(100 * (i + 1) / n_chunks)
                callback(progress, f"Processing chunk {i + 1}/{n_chunks}...")

            del chunk, result
            gc.collect()

        return output

    def cleanup(self) -> None:
        self.cache.release_all()


class SegmentationCache:
    """
    Shared cache for segmentation masks and metadata.

    Uses in-memory storage for small masks and memmap backend for large masks.
    """

    def __init__(self, cache_dir: Optional[str] = None, memmap_threshold_mb: int = 100):
        self._backend = MemmapStorageBackend(cache_dir=cache_dir, prefix="porous_seg")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._memmap_threshold_bytes = int(memmap_threshold_mb * 1024 * 1024)

    def _get_volume_id(self, raw_data: np.ndarray, threshold: int) -> str:
        return f"{id(raw_data)}_{threshold}"

    def has_cache(self, volume_id: str) -> bool:
        if volume_id not in self._cache:
            return False
        entry = self._cache[volume_id]
        if not entry.get("disk", False):
            return True
        return self._backend.exists(entry["backend_entry"])

    def store_pores_mask(
        self,
        volume_id: str,
        pores_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        use_disk: bool = True,
    ) -> None:
        self.clear(volume_id)

        cache_meta = metadata or {}
        if use_disk and pores_mask.nbytes > self._memmap_threshold_bytes:
            backend_entry = self._backend.write(volume_id, pores_mask)
            self._cache[volume_id] = {
                "disk": True,
                "backend_entry": backend_entry,
                "metadata": cache_meta,
            }
            print(f"[SegCache] Stored pores_mask to disk: {backend_entry['path']}")
            return

        self._cache[volume_id] = {
            "disk": False,
            "pores_mask": pores_mask.copy(),
            "metadata": cache_meta,
        }
        print(f"[SegCache] Stored pores_mask in memory: {volume_id}")

    def get_pores_mask(self, volume_id: str) -> Optional[np.ndarray]:
        entry = self._cache.get(volume_id)
        if entry is None:
            return None

        if entry.get("disk", False):
            return self._backend.read(entry["backend_entry"])
        return entry.get("pores_mask")

    def get_metadata(self, volume_id: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(volume_id)
        if entry is None:
            return None
        return entry.get("metadata")

    def clear(self, volume_id: Optional[str] = None) -> None:
        if volume_id is not None:
            entry = self._cache.pop(volume_id, None)
            if entry and entry.get("disk", False):
                self._backend.delete(entry["backend_entry"])
            gc.collect()
            return

        for entry in list(self._cache.values()):
            if entry.get("disk", False):
                self._backend.delete(entry["backend_entry"])
        self._cache.clear()
        gc.collect()
        print("[SegCache] Cleared cache")

    def __del__(self) -> None:
        try:
            self.clear()
        except Exception as exc:
            logger.debug("SegmentationCache cleanup during destruction failed: %s", exc)


class TimeSeriesPNMCache:
    """
    Cache for 4DCT time-series PNM tracking data persisted as pickle files.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._backend = PickleStorageBackend(cache_dir=cache_dir, prefix="porous_4dct_pnm")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def generate_key(self, volumes: List, threshold: int) -> str:
        volume_ids = []
        for vol in volumes:
            folder_name = vol.metadata.get("folder_name", "")
            vol_id = id(vol.raw_data) if vol.raw_data is not None else 0
            volume_ids.append(f"{folder_name}_{vol_id}")

        key_str = "_".join(volume_ids) + f"_t{threshold}"
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def has_cache(self, cache_key: str) -> bool:
        if cache_key not in self._cache:
            return False
        return self._backend.exists(self._cache[cache_key]["backend_entry"])

    def store(self, cache_key: str, time_series_pnm, reference_mesh, reference_snapshot) -> None:
        cache_data = {
            "time_series_pnm": time_series_pnm,
            "reference_mesh": reference_mesh,
            "reference_snapshot": reference_snapshot,
        }

        try:
            backend_entry = self._backend.write(cache_key, cache_data)
            self._cache[cache_key] = {
                "backend_entry": backend_entry,
                "num_timepoints": time_series_pnm.num_timepoints,
                "num_pores": time_series_pnm.num_reference_pores,
            }
            size_mb = os.path.getsize(backend_entry["path"]) / (1024 * 1024)
            print(
                f"[4DCT PNM Cache] Stored: {cache_key[:8]}... "
                f"({time_series_pnm.num_timepoints} timepoints, "
                f"{time_series_pnm.num_reference_pores} pores, {size_mb:.1f} MB)"
            )
        except Exception as exc:
            logger.error("Failed to store 4DCT PNM cache: %s", exc)

    def get(self, cache_key: str) -> Optional[Tuple[Any, Any, Any]]:
        entry = self._cache.get(cache_key)
        if entry is None:
            return None

        cache_data = self._backend.read(entry["backend_entry"])
        if cache_data is None:
            self._cache.pop(cache_key, None)
            return None

        print(
            f"[4DCT PNM Cache] Loaded: {cache_key[:8]}... "
            f"({entry['num_timepoints']} timepoints, {entry['num_pores']} pores)"
        )
        return (
            cache_data["time_series_pnm"],
            cache_data["reference_mesh"],
            cache_data["reference_snapshot"],
        )

    def clear(self, cache_key: Optional[str] = None) -> None:
        if cache_key is not None:
            entry = self._cache.pop(cache_key, None)
            if entry is not None:
                self._backend.delete(entry["backend_entry"])
            gc.collect()
            return

        for entry in list(self._cache.values()):
            self._backend.delete(entry["backend_entry"])
        self._cache.clear()
        gc.collect()
        print("[4DCT PNM Cache] Cleared all cache")

    def __del__(self) -> None:
        try:
            self.clear()
        except Exception as exc:
            logger.debug("TimeSeriesPNMCache cleanup during destruction failed: %s", exc)


_registry = CacheRegistry()


def get_disk_cache() -> DiskCacheManager:
    """Get the global disk cache manager."""
    return _registry.get("disk", DiskCacheManager)


def clear_disk_cache() -> None:
    """Clear the global disk cache manager."""
    _registry.clear("disk", drop_instance=True)


def get_segmentation_cache() -> SegmentationCache:
    """Get the global segmentation cache."""
    return _registry.get("segmentation", SegmentationCache)


def clear_segmentation_cache() -> None:
    """Clear the global segmentation cache."""
    _registry.clear("segmentation", drop_instance=False)


def get_timeseries_pnm_cache() -> TimeSeriesPNMCache:
    """Get the global time-series PNM cache."""
    return _registry.get("timeseries_pnm", TimeSeriesPNMCache)


def clear_timeseries_pnm_cache() -> None:
    """Clear the global time-series PNM cache."""
    _registry.clear("timeseries_pnm", drop_instance=False)


def clear_all_caches() -> None:
    """Clear all registered cache instances."""
    _registry.clear_all(drop_instances=True)
