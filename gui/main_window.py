"""
Main Application Window for Porous Media Analysis Suite.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, 
    QFrame, QSplitter, QScrollArea, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_POSITION, SPLITTER_SIZES
from gui.panels.info_panel import InfoPanel
from gui.panels.mode_panel import VisualizationModePanel
from gui.panels.params_panel import RenderingParametersPanel
from gui.panels.clip_panel import ClipPlanePanel
from gui.panels.roi_panel import ROIPanel
from gui.styles import PANEL_TITLE_STYLE


class MainWindow(QMainWindow):
    """
    Main Application Window.
    Handles UI layout, panel creation, and coordination between components.
    Delegates 3D rendering to a separate Renderer class.
    """
    
    def __init__(self):
        super().__init__()
        self.renderer = None
        self._data_manager = None
        
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(WINDOW_POSITION[0], WINDOW_POSITION[1], WINDOW_SIZE[0], WINDOW_SIZE[1])
        self._init_ui()
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_delayed_render)
        
        self.clip_update_timer = QTimer()
        self.clip_update_timer.setInterval(200)
        self.clip_update_timer.setSingleShot(True)
    
    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel
        left_scroll = self._create_control_panel()
        self.main_splitter.addWidget(left_scroll)
        
        # Center Panel (placeholder for plotter)
        self.plotter_container = QWidget()
        self.plotter_layout = QVBoxLayout(self.plotter_container)
        self.plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.addWidget(self.plotter_container)
        
        # Right Panel
        right_scroll = self._create_info_panel()
        self.main_splitter.addWidget(right_scroll)
        
        self.main_splitter.setSizes(SPLITTER_SIZES)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(2, False)
        
        main_layout.addWidget(self.main_splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready. Please load a sample scan.")
    
    def _create_control_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        panel = QWidget()
        self.control_panel_layout = QVBoxLayout(panel)
        layout = self.control_panel_layout
        layout.setSpacing(10)
        
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._create_separator())
        
        self.mode_panel = VisualizationModePanel()
        layout.addWidget(self.mode_panel)
        
        self.params_panel = RenderingParametersPanel()
        layout.addWidget(self.params_panel)
        
        self.clip_panel = ClipPlanePanel()
        layout.addWidget(self.clip_panel)
        
        self.roi_panel = ROIPanel()
        layout.addWidget(self.roi_panel)
        
        layout.addStretch()
        scroll_area.setWidget(panel)
        return scroll_area
    
    def _create_info_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        
        title = QLabel("Information")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.right_splitter = QSplitter(Qt.Vertical)
        self.info_panel = InfoPanel()
        self.right_splitter.addWidget(self.info_panel)
        
        layout.addWidget(self.right_splitter)
        scroll_area.setWidget(panel)
        return scroll_area
    
    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
    
    def set_renderer(self, renderer):
        self.renderer = renderer
        self._connect_signals()
    
    def set_plotter(self, plotter_widget):
        self.plotter_layout.addWidget(plotter_widget)
    
    def _connect_signals(self):
        if not self.renderer:
            return
        
        # Mode panel
        self.mode_panel.volume_clicked.connect(lambda: self.renderer.render_volume(reset_view=True))
        self.mode_panel.slices_clicked.connect(lambda: self.renderer.render_slices(reset_view=True))
        self.mode_panel.iso_clicked.connect(self.renderer.render_isosurface_auto)
        self.mode_panel.clear_clicked.connect(self.renderer.clear_view)
        self.mode_panel.reset_camera_clicked.connect(self.renderer.reset_camera)
        
        # Debounced signals
        for signal in [self.params_panel.solid_color_changed,
                       self.params_panel.light_angle_changed,
                       self.params_panel.coloring_mode_changed,
                       self.params_panel.render_style_changed,
                       self.params_panel.threshold_changed,
                       self.params_panel.slice_position_changed]:
            signal.connect(self.trigger_render)
        
        # Immediate updates
        self.params_panel.opacity_changed.connect(lambda: self.renderer.render_volume(reset_view=False))
        self.params_panel.clim_changed.connect(lambda: self.renderer.render_volume(reset_view=False))
        self.params_panel.colormap_changed.connect(self._on_colormap_changed)
        
        # Clip panel
        self.clip_panel.clip_toggled.connect(self.renderer._on_clip_toggled)
        self.clip_update_timer.timeout.connect(self.renderer._apply_clip_planes)
        self.clip_panel.clip_changed.connect(lambda: self.clip_update_timer.start())
        
        # ROI panel
        self.roi_panel.roi_toggled.connect(self.renderer._on_roi_toggled)
        self.roi_panel.apply_roi.connect(self.renderer._on_apply_roi)
        self.roi_panel.reset_roi.connect(self.renderer._on_reset_roi)
    
    def trigger_render(self):
        self.update_timer.start()
    
    def _perform_delayed_render(self):
        if not self.renderer:
            return
        mode = self.renderer.active_view_mode
        if mode == 'volume':
            self.renderer.render_volume(reset_view=False)
        elif mode == 'slices':
            self.renderer.render_slices(reset_view=False)
        elif mode == 'iso':
            self.renderer.render_isosurface_auto(reset_view=False)
        elif mode == 'mesh':
            self.renderer.render_mesh(reset_view=False)
    
    def _on_colormap_changed(self, text):
        if not self.renderer:
            return
        if self.renderer.active_view_mode == 'volume':
            self.renderer.render_volume(reset_view=False)
        else:
            self.trigger_render()
    
    def update_status(self, message: str):
        self.status_bar.showMessage(message)
    
    def add_custom_panel(self, panel: QWidget, index: int = 2, side: str = 'left'):
        if side == 'left' and hasattr(self, 'control_panel_layout'):
            self.control_panel_layout.insertWidget(index, panel)
        elif side == 'right' and hasattr(self, 'right_splitter'):
            self.right_splitter.addWidget(panel)
    
    def set_data_manager(self, data_manager):
        self._data_manager = data_manager
        if self.renderer:
            self.renderer.set_data_manager(data_manager)
