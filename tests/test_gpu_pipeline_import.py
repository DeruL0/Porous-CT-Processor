import importlib


def test_gpu_pipeline_module_is_importable():
    module = importlib.import_module("processors.gpu_pipeline")
    assert hasattr(module, "run_segmentation_pipeline_gpu")

