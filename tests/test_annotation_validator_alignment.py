from loaders.annotation_validator import AnnotationValidator


def test_validate_step_annotations_accepts_alignable_shape_mismatch():
    validator = AnnotationValidator(strict=False)
    result = validator._validate_step_annotations(
        step_index=0,
        volume_shape=[20, 30, 40],
        annotations={
            "step_index": 0,
            "volume_shape": [40, 30, 20],
            "voids": [],
            "num_voids": 0,
        },
        coco={"images": [], "annotations": [], "categories": []},
        label_stats={
            "shape": [40, 30, 20],
            "dtype": "int16",
            "unique_ids": [],
            "unique_ids_truncated": False,
            "nonzero_count": 0,
            "nonzero_count_estimated": False,
        },
        summary_step=None,
        summary_available=False,
    )

    assert result["ok"] is True
    assert result["checks"]["volume_shape_alignable"] is True
    assert result["checks"]["labels_shape_alignable"] is True
    assert not result["errors"]

