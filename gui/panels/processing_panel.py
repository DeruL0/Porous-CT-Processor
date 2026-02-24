"""
Structure Processing Panel for workflow actions.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.styles import PANEL_TITLE_STYLE
from gui.ui_constants import apply_group_layout, make_description_label, set_primary_button_policy
from config import THRESHOLD_RANGE_MIN, THRESHOLD_RANGE_MAX


class StructureProcessingPanel(QGroupBox):
    """
    Panel for Workflow Actions (Load, Process, Model, Export).
    """

    load_clicked = pyqtSignal()
    fast_load_clicked = pyqtSignal()
    dummy_clicked = pyqtSignal()
    extract_pores_clicked = pyqtSignal()
    auto_threshold_clicked = pyqtSignal()
    pnm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    gpu_toggled = pyqtSignal(bool)

    load_4dct_clicked = pyqtSignal()
    track_4dct_clicked = pyqtSignal()

    def __init__(self, title: str = "Structure Processing"):
        super().__init__()
        self.custom_title = title
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        apply_group_layout(layout)

        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(title_lbl)
        layout.addWidget(title_lbl)

        section_load = make_description_label(QLabel("--- Data Loading ---"))
        layout.addWidget(section_load)

        load_opt_layout = QHBoxLayout()
        self.load_4d_check = QCheckBox("Load as 4D Series")
        self.load_4d_check.setToolTip(
            "If checked, load buttons will ask for a folder containing timepoint subfolders."
        )
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Alphabetical", "Numeric", "Date Modified", "Manual"])
        self.sort_combo.setToolTip("Sorting order for 4D timepoints")
        self.sort_combo.setEnabled(False)
        self.load_4d_check.toggled.connect(self.sort_combo.setEnabled)
        load_opt_layout.addWidget(self.load_4d_check)
        load_opt_layout.addWidget(self.sort_combo, stretch=1)
        layout.addLayout(load_opt_layout)

        self._add_button(layout, "Load DICOM Series", self.load_clicked)
        self._add_button(layout, "Fast Load (2x Downsample)", self.fast_load_clicked)
        self._add_button(layout, "Generate Synthetic Sample", self.dummy_clicked)

        layout.addSpacing(8)
        section_process = make_description_label(QLabel("--- Segmentation & Modeling ---"))
        layout.addWidget(section_process)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(int(THRESHOLD_RANGE_MIN), int(THRESHOLD_RANGE_MAX))
        self.threshold_spin.setValue(-300)
        self.threshold_spin.setSingleStep(50)
        form.addRow("Threshold (Intensity):", self.threshold_spin)

        algo_row_widget = QWidget()
        algo_row_layout = QHBoxLayout(algo_row_widget)
        algo_row_layout.setContentsMargins(0, 0, 0, 0)
        algo_row_layout.setSpacing(8)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Auto", "Otsu", "Li", "Yen", "Triangle", "Minimum"])
        self.algo_combo.setToolTip(
            "Auto: Smart selection based on histogram\n"
            "Otsu: Classic bimodal thresholding\n"
            "Li: Minimum cross-entropy (noisy data)\n"
            "Yen: Maximum correlation\n"
            "Triangle: Good for CT peak+tail\n"
            "Minimum: Valley between peaks"
        )
        auto_btn = QPushButton("Detect")
        auto_btn.setToolTip("Calculate threshold using selected algorithm")
        auto_btn.clicked.connect(self.auto_threshold_clicked.emit)
        set_primary_button_policy(auto_btn)
        auto_btn.setMinimumWidth(80)
        auto_btn.setMaximumWidth(120)

        algo_row_layout.addWidget(self.algo_combo, stretch=1)
        algo_row_layout.addWidget(auto_btn)
        form.addRow("Algorithm:", algo_row_widget)
        layout.addLayout(form)

        from core.gpu_backend import is_gpu_available

        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setToolTip("Use CuPy/CUDA for accelerated processing")
        gpu_available = is_gpu_available()
        self.gpu_check.setChecked(gpu_available)
        self.gpu_check.setEnabled(gpu_available)
        self.gpu_check.toggled.connect(self.gpu_toggled.emit)
        layout.addWidget(self.gpu_check)

        layout.addSpacing(4)
        self._add_button(layout, "Extract Pores", self.extract_pores_clicked)
        self._add_button(layout, "Generate PNM Model", self.pnm_clicked)

        layout.addSpacing(8)
        self._add_button(layout, "Reset to Original", self.reset_clicked, min_height=35)
        self._add_button(layout, "Export VTK", self.export_clicked, min_height=35)

        self.setLayout(layout)

    def get_sort_mode(self) -> str:
        return self.sort_combo.currentText().lower()

    def _add_button(self, layout, text, signal, min_height: int = 40) -> None:
        btn = QPushButton(text)
        btn.setMinimumHeight(min_height)
        set_primary_button_policy(btn)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)

    def get_threshold(self) -> int:
        return self.threshold_spin.value()

    def set_threshold(self, value: int) -> None:
        self.threshold_spin.setValue(value)

    def get_algorithm(self) -> str:
        return self.algo_combo.currentText().lower()
