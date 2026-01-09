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
