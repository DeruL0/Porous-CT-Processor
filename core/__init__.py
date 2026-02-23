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
from core.coordinates import (
    raw_zyx_to_grid_xyz,
    world_xyz_to_voxel_zyx,
    world_xyz_to_index_zyx,
    world_delta_xyz_to_voxel_delta_zyx,
    voxel_delta_zyx_to_world_delta_xyz,
    bounds_xyz_to_slices_zyx,
    origin_xyz_for_subvolume_zyx,
    voxel_grid_zyx_to_world_xyz,
    voxel_zyx_to_world_xyz,
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
    'raw_zyx_to_grid_xyz', 'world_xyz_to_voxel_zyx', 'world_xyz_to_index_zyx',
    'world_delta_xyz_to_voxel_delta_zyx', 'voxel_delta_zyx_to_world_delta_xyz',
    'bounds_xyz_to_slices_zyx', 'origin_xyz_for_subvolume_zyx',
    'voxel_grid_zyx_to_world_xyz', 'voxel_zyx_to_world_xyz',
]
