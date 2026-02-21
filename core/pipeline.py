"""
Shared DAG pipeline used by both CLI and GUI workflows.

This module centralizes the core processing stages:
load -> threshold -> segment -> pnm -> export
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional
import inspect

from core.base import VolumeData
from core.chunker import DAGNode, SimpleDAGExecutor
from core.dto import VolumeProcessDTO
from core.progress import ProgressBus


PipelineStage = Literal["load", "threshold", "segment", "pnm", "export"]
PIPELINE_STAGE_ORDER: tuple[PipelineStage, ...] = ("load", "threshold", "segment", "pnm", "export")


def _noop_progress(_percent: int, _message: str) -> None:
    """Default no-op progress callback."""
    return


def resolve_pipeline_stages(
    target_stage: PipelineStage = "export",
    include_export: bool = True,
) -> tuple[PipelineStage, ...]:
    """
    Resolve the ordered list of stages required for a target stage.

    Example:
    - target_stage='segment' -> ('load', 'threshold', 'segment')
    - target_stage='pnm'     -> ('load', 'threshold', 'segment', 'pnm')
    """
    if target_stage not in PIPELINE_STAGE_ORDER:
        allowed = ", ".join(PIPELINE_STAGE_ORDER)
        raise ValueError(f"Unknown pipeline stage '{target_stage}'. Expected one of: {allowed}.")

    end_idx = PIPELINE_STAGE_ORDER.index(target_stage)
    stages = list(PIPELINE_STAGE_ORDER[: end_idx + 1])

    if not include_export and "export" in stages:
        stages.remove("export")

    return tuple(stages)


def _invoke_processor(processor: Any, data: VolumeData, progress: Callable[[int, str], None], **kwargs: Any) -> VolumeData:
    """
    Call processor.process while safely filtering kwargs not supported by signature.

    This prevents DTO drift from crashing runtime when optional parameters are
    present in DTO but not yet implemented in a processor.
    """
    sig = inspect.signature(processor.process)
    params = sig.parameters
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    call_kwargs: dict[str, Any] = {"callback": progress}
    for key, value in kwargs.items():
        if accepts_varkw or key in params:
            call_kwargs[key] = value

    return processor.process(data, **call_kwargs)


def _resolve_dummy_size(input_path: str) -> int:
    """
    Resolve DummyLoader size from input_path.

    Accepts an integer string (e.g. "96"), otherwise uses default 128.
    """
    if not input_path:
        return 128
    try:
        size = int(input_path)
        if size > 0:
            return size
    except (TypeError, ValueError):
        pass
    return 128


def _stage_load(dto: VolumeProcessDTO, progress: Callable[[int, str], None]) -> VolumeData:
    """Load raw data from input path."""
    progress(0, f"Loading input via {dto.loader_type}...")

    loader_type = (dto.loader_type or "dicom").lower()
    if loader_type == "dummy":
        from loaders import DummyLoader

        size = _resolve_dummy_size(dto.input_path)
        return DummyLoader().load(size=size, callback=progress)

    if loader_type == "dicom":
        from loaders import SmartDicomLoader

        if not dto.input_path:
            raise ValueError("input_path is required when loader_type='dicom'.")
        return SmartDicomLoader().load(dto.input_path, callback=progress)

    raise ValueError(f"Unknown loader_type: {dto.loader_type!r}. Supported: 'dicom', 'dummy'.")


def _stage_threshold(data: VolumeData, dto: VolumeProcessDTO, progress: Callable[[int, str], None]) -> float:
    """Compute threshold value (fixed or auto)."""
    if data.raw_data is None:
        raise ValueError("Threshold stage requires voxel data (raw_data).")

    if dto.auto_threshold:
        progress(0, "Detecting threshold (auto)...")
        from processors import PoreExtractionProcessor

        threshold = float(PoreExtractionProcessor.suggest_threshold(data, algorithm="auto"))
        progress(100, f"Auto threshold: {threshold:.1f}")
        return threshold

    threshold = float(dto.threshold)
    progress(100, f"Using fixed threshold: {threshold:.1f}")
    return threshold


def _stage_segment(
    data: VolumeData,
    threshold: float,
    dto: VolumeProcessDTO,
    progress: Callable[[int, str], None],
) -> VolumeData:
    """Segment void space from solid matrix."""
    from processors import PoreExtractionProcessor

    processor = PoreExtractionProcessor()
    return _invoke_processor(
        processor=processor,
        data=data,
        progress=progress,
        threshold=int(threshold),
        min_pore_size=dto.min_pore_size,
    )


def _stage_pnm(
    segmented: VolumeData,
    threshold: float,
    dto: VolumeProcessDTO,
    progress: Callable[[int, str], None],
) -> VolumeData:
    """Generate Pore Network Model from segmented data."""
    from processors import PoreToSphereProcessor

    processor = PoreToSphereProcessor()
    return _invoke_processor(
        processor=processor,
        data=segmented,
        progress=progress,
        threshold=int(threshold),
        min_throat_dist=dto.min_throat_dist,
    )


def _stage_export(
    data_dict: dict[str, Any],
    dto: VolumeProcessDTO,
    progress: Callable[[int, str], None],
) -> list[str]:
    """Export requested outputs and return exported file paths."""
    out_dir = Path(dto.output_dir) if dto.output_dir else Path.cwd() / "cli_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    segmented = data_dict.get("segment")
    pnm = data_dict.get("pnm")
    formats = tuple(fmt.lower() for fmt in dto.export_formats)

    exported: list[str] = []
    tasks: list[tuple[str, Any, Path]] = []

    if "vtk" in formats:
        if pnm is not None:
            tasks.append(("vtk", pnm, out_dir / "pnm_mesh.vtp"))
        if segmented is not None and getattr(segmented, "raw_data", None) is not None:
            tasks.append(("vtk", segmented, out_dir / "segmented.vti"))
    if "tiff" in formats and segmented is not None and getattr(segmented, "raw_data", None) is not None:
        tasks.append(("tiff", segmented, out_dir / "segmented.tif"))
    if "npy" in formats and segmented is not None and getattr(segmented, "raw_data", None) is not None:
        tasks.append(("npy", segmented, out_dir / "segmented.npy"))

    if not tasks:
        progress(100, "No export tasks requested.")
        return exported

    from exporters import VTKExporter

    total = len(tasks)
    for idx, (fmt, obj, path) in enumerate(tasks, start=1):
        base_pct = int(100 * (idx - 1) / total)
        progress(base_pct, f"Exporting {fmt.upper()} -> {path.name}")

        if fmt == "vtk":
            VTKExporter.export(obj, str(path))
        elif fmt == "tiff":
            try:
                from tifffile import imwrite
            except ImportError:
                continue
            imwrite(str(path), obj.raw_data.astype("uint8"))
        elif fmt == "npy":
            import numpy as np

            np.save(str(path), obj.raw_data)
        else:
            continue

        exported.append(str(path))

    progress(100, "Export complete.")
    return exported


def build_volume_pipeline(
    dto: VolumeProcessDTO,
    *,
    input_data: Optional[VolumeData] = None,
    target_stage: PipelineStage = "export",
    include_export: bool = True,
    progress_bus: Optional[ProgressBus] = None,
    stage_progress_factory: Optional[Callable[[PipelineStage], Callable[[int, str], None]]] = None,
) -> SimpleDAGExecutor:
    """
    Build a DAG executor for the standard volume processing pipeline.

    Args:
        dto: Pipeline configuration.
        input_data: Optional preloaded data. If provided, load stage returns this object.
        target_stage: Last stage to execute.
        include_export: Whether to include export stage if target allows it.
        progress_bus: Optional progress event bus.
        stage_progress_factory: Optional per-stage progress callback factory.
    """
    stages = resolve_pipeline_stages(target_stage=target_stage, include_export=include_export)
    dag = SimpleDAGExecutor()

    def stage_progress(stage: PipelineStage) -> Callable[[int, str], None]:
        if stage_progress_factory is None:
            if progress_bus is None:
                return _noop_progress
            return progress_bus.stage_callback(stage)
        return stage_progress_factory(stage)

    if "load" in stages:
        dag.add(
            DAGNode(
                name="load",
                fn=lambda _deps: input_data if input_data is not None else _stage_load(dto, stage_progress("load")),
                depends_on=(),
            )
        )

    if "threshold" in stages:
        dag.add(
            DAGNode(
                name="threshold",
                fn=lambda deps: _stage_threshold(deps["load"], dto, stage_progress("threshold")),
                depends_on=("load",),
            )
        )

    if "segment" in stages:
        dag.add(
            DAGNode(
                name="segment",
                fn=lambda deps: _stage_segment(deps["load"], deps["threshold"], dto, stage_progress("segment")),
                depends_on=("load", "threshold"),
            )
        )

    if "pnm" in stages:
        dag.add(
            DAGNode(
                name="pnm",
                fn=lambda deps: _stage_pnm(deps["segment"], deps["threshold"], dto, stage_progress("pnm")),
                depends_on=("segment", "threshold"),
            )
        )

    if "export" in stages:
        dag.add(
            DAGNode(
                name="export",
                fn=lambda deps: _stage_export(deps, dto, stage_progress("export")),
                depends_on=("segment", "pnm"),
            )
        )

    return dag


def run_volume_pipeline(
    dto: VolumeProcessDTO,
    *,
    input_data: Optional[VolumeData] = None,
    target_stage: PipelineStage = "export",
    include_export: bool = True,
    progress_bus: Optional[ProgressBus] = None,
    dag_progress: Optional[Callable[[int, str], None]] = None,
    stage_progress_factory: Optional[Callable[[PipelineStage], Callable[[int, str], None]]] = None,
) -> dict[str, Any]:
    """
    Execute the shared volume pipeline and return stage outputs keyed by stage name.
    """
    dag = build_volume_pipeline(
        dto=dto,
        input_data=input_data,
        target_stage=target_stage,
        include_export=include_export,
        progress_bus=progress_bus,
        stage_progress_factory=stage_progress_factory,
    )
    return dag.run(progress=dag_progress)


__all__ = [
    "PipelineStage",
    "PIPELINE_STAGE_ORDER",
    "resolve_pipeline_stages",
    "build_volume_pipeline",
    "run_volume_pipeline",
]
