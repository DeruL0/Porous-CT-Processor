"""
Structure Processing Panel for workflow actions.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QHBoxLayout
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
    auto_threshold_clicked = pyqtSignal()
    pnm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    gpu_toggled = pyqtSignal(bool)

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
        
        # Threshold Input
        thresh_layout = QHBoxLayout()
        thresh_lbl = QLabel("Threshold (HU):")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(-10000, 30000)
        self.threshold_spin.setValue(-300)  # Default for air
        self.threshold_spin.setSingleStep(50)
        
        thresh_layout.addWidget(thresh_lbl)
        thresh_layout.addWidget(self.threshold_spin)
        layout.addLayout(thresh_layout)
        
        # Algorithm Selection + Auto Button
        from PyQt5.QtWidgets import QComboBox
        algo_layout = QHBoxLayout()
        algo_lbl = QLabel("Algorithm:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Auto", "Otsu", "Li", "Yen", "Triangle", "Minimum"])
        self.algo_combo.setToolTip(
            "Auto: Smart selection based on histogram\n"
            "Otsu: Classic bimodal thresholding\n"
            "Li: Minimum cross-entropy (noisy data)\n"
            "Yen: Maximum correlation\n"
            "Triangle: Good for CT 'peak+tail'\n"
            "Minimum: Valley between peaks"
        )
        
        # Auto Button
        auto_btn = QPushButton("Detect")
        auto_btn.setFixedWidth(55)
        auto_btn.setToolTip("Calculate threshold using selected algorithm")
        auto_btn.clicked.connect(self.auto_threshold_clicked.emit)
        
        algo_layout.addWidget(algo_lbl)
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addWidget(auto_btn)
        layout.addLayout(algo_layout)
        
        # GPU Toggle
        from PyQt5.QtWidgets import QCheckBox
        from core.gpu_backend import is_gpu_available
        
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setToolTip("Use CuPy/CUDA for accelerated processing")
        # Check if actually available (library installed + device present)
        is_available = is_gpu_available()
        self.gpu_check.setChecked(is_available)
        self.gpu_check.setEnabled(is_available)
        self.gpu_check.toggled.connect(self.gpu_toggled.emit)
        
        layout.addWidget(self.gpu_check)
        
        layout.addSpacing(5)
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

    def get_threshold(self) -> int:
        return self.threshold_spin.value()

    def set_threshold(self, value: int):
        self.threshold_spin.setValue(value)

    def get_algorithm(self) -> str:
        """Get selected threshold algorithm (auto, otsu, li, yen, triangle, minimum)."""
        return self.algo_combo.currentText().lower()
