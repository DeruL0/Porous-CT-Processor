"""
Structure Processing Panel for workflow actions.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox
from PyQt5.QtCore import pyqtSignal

from gui.styles import PANEL_TITLE_STYLE


class StructureProcessingPanel(QGroupBox):
    """
    Panel for Workflow Actions (Load, Process, Model, Export).
    """
    # Signals
    load_clicked = pyqtSignal()
    fast_load_clicked = pyqtSignal()
    dummy_clicked = pyqtSignal()
    extract_pores_clicked = pyqtSignal()
    pnm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    export_clicked = pyqtSignal()

    def __init__(self, title: str = "🔧 Structure Processing"):
        super().__init__()
        self.custom_title = title
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        layout.addWidget(title_lbl)
        
        # Data Loading Section
        self._add_button(layout, "📂 Load DICOM Series", self.load_clicked)
        self._add_button(layout, "⚡ Fast Load (2x Downsample)", self.fast_load_clicked)
        self._add_button(layout, "🧪 Generate Synthetic Sample", self.dummy_clicked)
        
        # Processing Section
        layout.addSpacing(10)
        self._add_button(layout, "🔬 Extract Pores", self.extract_pores_clicked)
        self._add_button(layout, "🌐 Generate PNM Model", self.pnm_clicked)
        
        # Reset Section
        layout.addSpacing(10)
        self._add_button(layout, "↩ Reset to Original", self.reset_clicked, min_height=35)
        self._add_button(layout, "💾 Export VTK", self.export_clicked, min_height=35)
        
        self.setLayout(layout)

    def _add_button(self, layout, text, signal, min_height=40):
        btn = QPushButton(text)
        btn.setMinimumHeight(min_height)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)
