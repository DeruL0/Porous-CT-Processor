"""
Configuration constants for the Porous Media Analysis Suite.
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

# ==========================================
# Processing
# ==========================================
MIN_PEAK_DISTANCE = 6
DEFAULT_THRESHOLD_PORE = -300

# ==========================================
# Timer Intervals (ms)
# ==========================================
DEBOUNCE_INTERVAL = 100
CLIP_UPDATE_INTERVAL = 200
