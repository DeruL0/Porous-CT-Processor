"""
Configuration constants for the Porous Media Analysis Suite.
All thresholds and configurable parameters are centralized here.
"""

# ==========================================
# Window Settings
# ==========================================
WINDOW_TITLE = "Porous Media Analysis Suite"
WINDOW_SIZE = (1400, 900)
WINDOW_POSITION = (100, 100)

# Splitter sizes (left:center:right)
SPLITTER_SIZES = [350, 900, 350]

# ==========================================
# Loader Settings
# ==========================================

# SmartDicomLoader auto-strategy thresholds (number of files)
LOADER_THRESHOLD_FAST = 200       # >200 files → use FastDicomLoader
LOADER_THRESHOLD_MMAP = 500       # >500 files → use MemoryMappedDicomLoader  
LOADER_THRESHOLD_CHUNKED = 1000   # >1000 files → use ChunkedDicomLoader

# Parallel reading settings
LOADER_MAX_WORKERS = 4            # Number of parallel threads for file reading
LOADER_DOWNSAMPLE_STEP = 2        # Downsample step for FastDicomLoader

# ChunkedDicomLoader settings
LOADER_CHUNK_SIZE = 64            # Number of slices per chunk

# ==========================================
# Rendering Settings
# ==========================================

# Memory thresholds for auto-downsampling (number of voxels)
RENDER_MAX_VOXELS_VOLUME = 100_000_000    # 100M voxels for volume rendering
RENDER_MAX_VOXELS_ISO = 150_000_000       # 150M voxels for isosurface
RENDER_MAX_MEMORY_MB = 500                # Max memory budget (MB) for render grid

# LOD (Level of Detail) settings
LOD_LEVELS = 3                    # Number of LOD levels to create
LOD_FAR_THRESHOLD = 1000          # Camera distance for lowest LOD
LOD_MID_THRESHOLD = 500           # Camera distance for medium LOD

# ==========================================
# Processing Settings (Pore Extraction)
# ==========================================

# Chunked processing threshold (bytes)
PROCESS_CHUNK_THRESHOLD = 500 * 1024 * 1024  # 500 MB - use chunked above this
PROCESS_CHUNK_SIZE = 32           # Slices per chunk for pore processing

# Otsu sampling threshold (number of voxels)
PROCESS_OTSU_SAMPLE_THRESHOLD = 100_000_000  # Sample data above this size

# ==========================================
# Colormaps
# ==========================================
DEFAULT_COLORMAPS = [
    'bone', 'viridis', 'plasma', 'gray', 
    'coolwarm', 'jet', 'magma'
]

DEFAULT_COLORMAP = 'bone'

# ==========================================
# Solid Colors
# ==========================================
SOLID_COLORS = [
    'ivory', 'red', 'gold', 'lightgray', 
    'mediumseagreen', 'dodgerblue', 'wheat'
]

DEFAULT_SOLID_COLOR = 'ivory'

# ==========================================
# Volume Rendering
# ==========================================
OPACITY_PRESETS = [
    'sigmoid', 'sigmoid_10', 'linear', 
    'linear_r', 'geom', 'geom_r'
]

DEFAULT_OPACITY = 'sigmoid'

# ==========================================
# Isosurface Rendering
# ==========================================
RENDER_STYLES = ['Surface', 'Wireframe', 'Wireframe + Surface']
COLORING_MODES = ['Solid Color', 'Depth (Z-Axis)', 'Radial (Center Dist)']

DEFAULT_THRESHOLD = 300
DEFAULT_THRESHOLD_PORE = -300

# ==========================================
# PNM Processing
# ==========================================
MIN_PEAK_DISTANCE = 6

# ==========================================
# Timer Intervals (ms)
# ==========================================
DEBOUNCE_INTERVAL = 100
CLIP_UPDATE_INTERVAL = 200

# ==========================================
# GPU Acceleration Settings
# ==========================================
GPU_ENABLED = True    # Set False to disable GPU acceleration
GPU_MIN_SIZE_MB = 10  # Minimum data size (MB) to use GPU (smaller uses CPU)

# ==========================================
# 4D CT Tracking Settings
# ==========================================
TRACKING_IOU_THRESHOLD = 0.1           # Min IoU to consider pores matched
TRACKING_COMPRESSION_THRESHOLD = 0.01  # Volume ratio below which pore is "compressed"
TRACKING_ANIMATION_FPS = 5             # Animation playback speed

# Tracking algorithm options
TRACKING_USE_GPU = True                # Use GPU acceleration for tracking (requires CuPy)
TRACKING_USE_BATCH = True              # Use batch IoU calculation (faster for many pores)
TRACKING_USE_HUNGARIAN = False         # Use Hungarian algorithm for global optimal matching
TRACKING_BATCH_SIZE = 1000             # Max pores to process in one batch (memory vs speed)
TRACKING_GPU_MIN_PORES = 100           # Min pores to use GPU (smaller uses CPU)

# TGGA matching options
TRACKING_MATCH_MODE = "temporal_global"      # temporal_global | legacy_greedy | global_iou_legacy
TRACKING_ASSIGN_SOLVER = "lapjv"             # lapjv | scipy
TRACKING_COST_WEIGHTS = (0.45, 0.30, 0.20, 0.05)  # IoU, center distance, volume ratio, dice
TRACKING_VOLUME_COST_MODE = "symdiff"        # symdiff | gaussian
TRACKING_VOLUME_COST_GAUSSIAN_SIGMA = 25.0   # Used only when TRACKING_VOLUME_COST_MODE="gaussian"
TRACKING_MAX_MISSES = 2
TRACKING_GATE_CENTER_RADIUS_FACTOR = 2.5
TRACKING_GATE_VOLUME_RATIO_MIN = 1e-4
TRACKING_GATE_VOLUME_RATIO_MAX = 5.0
TRACKING_GATE_IOU_MIN = 0.02
TRACKING_GATE_VOLUME_RATIO_MIN_FLOOR = 1e-4

# Macro pre-registration (DVC-style) before local gating
TRACKING_ENABLE_MACRO_REGISTRATION = True
TRACKING_MACRO_REG_SMOOTHING_SIGMA = 1.5
TRACKING_MACRO_REG_UPSAMPLE_FACTOR = 4
TRACKING_MACRO_REG_USE_GPU = True
TRACKING_MACRO_REG_GPU_MIN_MB = 8.0

# Constant-acceleration Kalman predictor
TRACKING_KALMAN_PROCESS_NOISE = 0.05
TRACKING_KALMAN_MEASUREMENT_NOISE = 1.0
TRACKING_KALMAN_BRAKE_VELOCITY_DECAY = 0.75
TRACKING_KALMAN_BRAKE_ACCEL_DECAY = 0.35
TRACKING_KALMAN_FREEZE_AFTER_MISSES = 3

# Pore closure event decision
TRACKING_CLOSURE_VOLUME_RATIO_THRESHOLD = 0.02
TRACKING_CLOSURE_MIN_VOLUME_VOXELS = 2.0
TRACKING_CLOSURE_STRAIN_THRESHOLD = 0.6

# Position smoothing to reduce center jitter across timepoints (0 = lock to previous, 1 = follow current)
TRACKING_CENTER_SMOOTHING = 0.35

# Topology preservation
TRACKING_PRESERVE_TOPOLOGY = True      # Keep reference connectivity (t=0) for all timepoints

