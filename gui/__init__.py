"""
GUI Package for Porous Media Analysis Suite.
Exports all panel classes and MainWindow.
"""

from gui.styles import (
    PANEL_TITLE_STYLE, 
    TABLE_STYLESHEET,
    PRIMARY_BUTTON_STYLE,
    SECONDARY_BUTTON_STYLE
)

from gui.panels.info_panel import InfoPanel, StatisticsPanel
from gui.panels.mode_panel import VisualizationModePanel
from gui.panels.params_panel import RenderingParametersPanel
from gui.panels.clip_panel import ClipPlanePanel
from gui.panels.roi_panel import ROIPanel
from gui.panels.processing_panel import StructureProcessingPanel
from gui.panels.timeseries_panel import TimeSeriesControlPanel, TrackingAnalysisPanel
from gui.main_window import MainWindow

__all__ = [
    # Styles
    'PANEL_TITLE_STYLE',
    'TABLE_STYLESHEET',
    'PRIMARY_BUTTON_STYLE',
    'SECONDARY_BUTTON_STYLE',
    # Panels
    'InfoPanel',
    'StatisticsPanel',
    'VisualizationModePanel',
    'RenderingParametersPanel',
    'ClipPlanePanel',
    'ROIPanel',
    'StructureProcessingPanel',
    'TimeSeriesControlPanel',
    'TrackingAnalysisPanel',
    # Window
    'MainWindow',
]
