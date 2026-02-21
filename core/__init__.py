"""
Core module containing base classes and data structures.
"""

from core.base import VolumeData, BaseLoader, BaseProcessor, BaseVisualizer
from core.gpu_backend import get_gpu_backend, is_gpu_available
from core.time_series import PNMSnapshot, PoreTrackingResult, TimeSeriesPNM, PoreStatus
from core.dto import RenderParamsDTO, VolumeProcessDTO
from core.chunker import SpatialChunker, ChunkDescriptor, DAGNode, SimpleDAGExecutor, edt_chunked
from core.progress import (
    ProgressEvent,
    ProgressObserver,
    ProgressBus,
    StageProgressMapper,
    CancelFlagObserver,
    TerminalProgressObserver,
)
from core.pipeline import (
    PipelineStage,
    PIPELINE_STAGE_ORDER,
    resolve_pipeline_stages,
    build_volume_pipeline,
    run_volume_pipeline,
)

__all__ = [
    'VolumeData', 'BaseLoader', 'BaseProcessor', 'BaseVisualizer',
    'get_gpu_backend', 'is_gpu_available',
    'PNMSnapshot', 'PoreTrackingResult', 'TimeSeriesPNM', 'PoreStatus',
    'RenderParamsDTO', 'VolumeProcessDTO',
    'SpatialChunker', 'ChunkDescriptor', 'DAGNode', 'SimpleDAGExecutor', 'edt_chunked',
    'ProgressEvent', 'ProgressObserver', 'ProgressBus', 'StageProgressMapper',
    'CancelFlagObserver', 'TerminalProgressObserver',
    'PipelineStage', 'PIPELINE_STAGE_ORDER', 'resolve_pipeline_stages',
    'build_volume_pipeline', 'run_volume_pipeline',
]
