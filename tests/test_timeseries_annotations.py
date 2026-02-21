import json
from pathlib import Path

import numpy as np

from core import VolumeData
from loaders import AnnotationValidator
from loaders.time_series import TimeSeriesDicomLoader


class _DummyVolumeLoader:
    def __init__(self, shape_by_folder):
        self._shape_by_folder = dict(shape_by_folder)

    def load(self, folder_path, callback=None):
        folder_name = Path(folder_path).name
        shape = self._shape_by_folder[folder_name]
        return VolumeData(
            raw_data=np.zeros(shape, dtype=np.int16),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            metadata={"Type": "Dummy"},
        )


def _write_step_assets(step_dir: Path, step_index: int, volume_shape, labels_shape=None):
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "CT_0000.dcm").write_bytes(b"DICM")

    labels_shape = labels_shape or volume_shape
    labels = np.zeros(labels_shape, dtype=np.int16)
    labels[(0,) * len(labels_shape)] = 1
    np.save(step_dir / "labels.npy", labels)

    annotations = {
        "step_index": step_index,
        "compression_ratio": float(step_index) * 0.1,
        "voxel_size": 0.5,
        "volume_shape": list(volume_shape),
        "origin": [0.0, 0.0, 0.0],
        "num_voids": 1,
        "voids": [
            {
                "id": 1,
                "shape": "sphere",
                "center_mm": [1.0, 1.0, 1.0],
                "radius_mm": 0.8,
                "volume_mm3": 2.1,
                "bbox_voxel_min": [0, 0, 0],
                "bbox_voxel_max": [1, 1, 1],
                "center_voxel": [0.0, 0.0, 0.0],
            }
        ],
    }
    (step_dir / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

    coco = {
        "images": [{"id": 1, "file_name": "slice_0000.png", "width": volume_shape[2], "height": volume_shape[1]}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
                "void_id": 1,
                "void_shape": "sphere",
            }
        ],
        "categories": [{"id": 1, "name": "void", "supercategory": "defect"}],
    }
    (step_dir / "coco_annotations.json").write_text(json.dumps(coco), encoding="utf-8")


def _write_summary(parent: Path, num_steps: int):
    summary = {
        "num_steps": num_steps,
        "config": {"total_compression": 0.2, "num_steps": max(0, num_steps - 1), "axis": "Z"},
        "steps": [
            {"step_index": i, "compression_ratio": i * 0.1, "num_voids": 1}
            for i in range(num_steps)
        ],
    }
    (parent / "annotations_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_timeseries_loader_reads_sim_annotations(tmp_path):
    shape = (12, 10, 8)
    _write_step_assets(tmp_path / "Step_00", step_index=0, volume_shape=shape)
    _write_step_assets(tmp_path / "Step_01", step_index=1, volume_shape=shape)
    _write_summary(tmp_path, num_steps=2)

    loader = TimeSeriesDicomLoader(loader=_DummyVolumeLoader({"Step_00": shape, "Step_01": shape}))
    volumes = loader.load_series(str(tmp_path), sort_mode="alphabetical")
    for volume in volumes:
        assert "sim_annotations" not in volume.metadata

    report = AnnotationValidator(strict=False).validate_series(parent_folder=str(tmp_path), volumes=volumes)

    assert len(volumes) == 2
    for i, volume in enumerate(volumes):
        assert "sim_annotations" in volume.metadata
        sim = volume.metadata["sim_annotations"]
        assert sim["annotations"]["step_index"] == i
        assert sim["labels"]["shape"] == list(shape)
        assert sim["validation"]["ok"] is True

        series = volume.metadata["sim_annotation_series"]
        assert series["summary"]["available"] is True
        assert series["validation"]["error_count"] == 0
    assert report["validation"]["ok"] is True


def test_timeseries_loader_reports_label_shape_mismatch(tmp_path):
    shape = (10, 10, 10)
    _write_step_assets(tmp_path / "Step_00", step_index=0, volume_shape=shape, labels_shape=(6, 6, 6))
    _write_summary(tmp_path, num_steps=1)

    loader = TimeSeriesDicomLoader(loader=_DummyVolumeLoader({"Step_00": shape}))
    volumes = loader.load_series(str(tmp_path), sort_mode="alphabetical")
    AnnotationValidator(strict=False).validate_series(parent_folder=str(tmp_path), volumes=volumes)

    sim = volumes[0].metadata["sim_annotations"]
    assert sim["validation"]["ok"] is False
    assert any("labels.npy shape" in msg for msg in sim["validation"]["errors"])
    assert volumes[0].metadata["sim_annotation_series"]["validation"]["error_count"] >= 1
