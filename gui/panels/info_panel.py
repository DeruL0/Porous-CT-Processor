"""
Information panels for metadata and statistics display.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.styles import PANEL_TITLE_STYLE, TABLE_STYLESHEET
from gui.ui_constants import TABLE_MIN_HEIGHT, apply_group_layout, apply_panel_layout, make_description_label


class StatisticsPanel(QWidget):
    """
    Panel to display derived statistics and histograms for PNM analysis.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        title = QLabel("Statistics & Analysis")
        title.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(title)
        layout.addWidget(title)

        self.tabs = QTabWidget()
        # Keep chart area compact by default to avoid overly long bars/plots.
        self.tabs.setMinimumHeight(210)
        self.tabs.setMaximumHeight(280)
        self.tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.fig_pore = Figure(figsize=(5, 3), dpi=80)
        self.canvas_pore = FigureCanvas(self.fig_pore)
        self.tabs.addTab(self.canvas_pore, "Pore Sizes")

        self.fig_throat = Figure(figsize=(5, 3), dpi=80)
        self.canvas_throat = FigureCanvas(self.fig_throat)
        self.tabs.addTab(self.canvas_throat, "Throat Sizes")
        layout.addWidget(self.tabs)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setStyleSheet(TABLE_STYLESHEET)
        self.stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats_table.setMinimumHeight(TABLE_MIN_HEIGHT)
        layout.addWidget(self.stats_table)
        layout.addStretch(1)

    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400) -> None:
        table.doItemsLayout()
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        height += 6
        clamped_height = max(TABLE_MIN_HEIGHT, min(height, max_height))
        table.setMinimumHeight(clamped_height)
        table.setMaximumHeight(max_height)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if height > max_height else Qt.ScrollBarAlwaysOff
        )

    def update_statistics(self, metadata) -> None:
        self.fig_pore.clear()
        self.fig_throat.clear()
        self.stats_table.setRowCount(0)

        pore_dist = metadata.get("PoreSizeDistribution", {})
        throat_dist = metadata.get("ThroatSizeDistribution", {})
        largest_pore = metadata.get("LargestPoreRatio", "N/A")
        throat_stats = metadata.get("ThroatStats", {})
        pore_count = metadata.get("PoreCount", 0)
        connection_count = metadata.get("ConnectionCount", 0)

        if pore_dist.get("bins") and pore_dist.get("counts"):
            ax = self.fig_pore.add_subplot(111)
            bins = np.array(pore_dist["bins"])
            counts = pore_dist["counts"]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bins) * 0.85,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xlabel("Pore Radius (mm)")
            ax.set_ylabel("Count")
            ax.set_title("Pore Size Distribution")
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.02)
            self.fig_pore.tight_layout()

        if throat_dist.get("bins") and throat_dist.get("counts"):
            ax = self.fig_throat.add_subplot(111)
            bins = np.array(throat_dist["bins"])
            counts = throat_dist["counts"]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bins) * 0.85,
                alpha=0.7,
                color="indianred",
                edgecolor="black",
            )
            ax.set_xlabel("Throat Radius (mm)")
            ax.set_ylabel("Count")
            ax.set_title("Throat Size Distribution")
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.02)
            self.fig_throat.tight_layout()

        self.canvas_pore.draw()
        self.canvas_throat.draw()

        stats_data = [
            ("Total Pores", str(pore_count)),
            ("Connections", str(connection_count)),
            ("Largest Pore %", str(largest_pore)),
        ]
        if throat_stats:
            stats_data.extend(
                [
                    ("Throat Min", f"{throat_stats.get('min', 0):.3f} mm"),
                    ("Throat Max", f"{throat_stats.get('max', 0):.3f} mm"),
                    ("Throat Mean", f"{throat_stats.get('mean', 0):.3f} mm"),
                ]
            )

        self.stats_table.setRowCount(len(stats_data))
        for row, (metric, value) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))
        self._adjust_table_height(self.stats_table, max_height=350)

    def clear(self) -> None:
        self.fig_pore.clear()
        self.fig_throat.clear()
        self.canvas_pore.draw()
        self.canvas_throat.draw()
        self.stats_table.setRowCount(0)


class InfoPanel(QGroupBox):
    """
    Panel to display source metadata and computed metrics.
    """

    def __init__(self, title: str = "Sample Information"):
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

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.info_table.setStyleSheet(TABLE_STYLESHEET)
        self.info_table.setShowGrid(True)
        self.info_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.info_table.setMinimumHeight(TABLE_MIN_HEIGHT)

        self.computed_label = QLabel("<b>Computed Metrics</b>")
        self.computed_label.setStyleSheet(PANEL_TITLE_STYLE)
        make_description_label(self.computed_label)

        self.computed_table = QTableWidget()
        self.computed_table.setColumnCount(2)
        self.computed_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.computed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.computed_table.verticalHeader().setVisible(False)
        self.computed_table.setAlternatingRowColors(True)
        self.computed_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.computed_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.computed_table.setStyleSheet(TABLE_STYLESHEET)
        self.computed_table.setShowGrid(True)
        self.computed_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.computed_table.setMinimumHeight(TABLE_MIN_HEIGHT)

        layout.addWidget(self.info_table)
        layout.addWidget(self.computed_label)
        layout.addWidget(self.computed_table)
        self.setLayout(layout)

    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400) -> None:
        table.doItemsLayout()
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        height += 6
        clamped_height = max(TABLE_MIN_HEIGHT, min(height, max_height))
        table.setMinimumHeight(clamped_height)
        table.setMaximumHeight(max_height)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if height > max_height else Qt.ScrollBarAlwaysOff
        )

    def update_info(self, type_str: str, dim: tuple, spacing: tuple, metadata: dict) -> None:
        self.info_table.setRowCount(0)
        self.computed_table.setRowCount(0)

        basic_data = [("Data Type", type_str)]
        if dim != (0, 0, 0):
            basic_data.append(("Dimensions", f"{dim[0]} x {dim[1]} x {dim[2]}"))
        elif "MeshPoints" in metadata:
            basic_data.append(("Mesh Points", str(metadata["MeshPoints"])))
        basic_data.append(("Voxel Size", f"{spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm"))

        for key in ("Porosity", "PoreCount", "ConnectionCount"):
            if key in metadata:
                val = metadata[key]
                if isinstance(val, float) and key == "Porosity":
                    val = f"{val:.2f}%"
                basic_data.append((key, str(val)))

        self.info_table.setRowCount(len(basic_data))
        for row, (prop, val) in enumerate(basic_data):
            self.info_table.setItem(row, 0, QTableWidgetItem(prop))
            self.info_table.setItem(row, 1, QTableWidgetItem(val))
        self._adjust_table_height(self.info_table, max_height=300)

        computed = []
        if dim != (0, 0, 0):
            total_voxels = dim[0] * dim[1] * dim[2]
            computed.append(("Total Voxels", f"{total_voxels:,}"))
            phys_volume = dim[0] * spacing[0] * dim[1] * spacing[1] * dim[2] * spacing[2]
            computed.append(("Physical Volume", f"{phys_volume / 1000:.2f} cm3" if phys_volume > 1000 else f"{phys_volume:.2f} mm3"))
            phys_dim = (dim[0] * spacing[0], dim[1] * spacing[1], dim[2] * spacing[2])
            computed.append(("Physical Size", f"{phys_dim[0]:.1f} x {phys_dim[1]:.1f} x {phys_dim[2]:.1f} mm"))
            mem_mb = (total_voxels * 4) / (1024 * 1024)
            computed.append(("Memory (Est.)", f"{mem_mb:.1f} MB"))

        if "ThroatStats" in metadata:
            ts = metadata["ThroatStats"]
            if "mean" in ts:
                computed.append(("Avg Throat Radius", f"{ts['mean']:.3f} mm"))
            if "min" in ts and "max" in ts:
                computed.append(("Throat Range", f"{ts['min']:.3f} - {ts['max']:.3f} mm"))

        for key, label in (
            ("LargestPoreRatio", "Largest Pore %"),
            ("Permeability_mD", "Permeability"),
            ("Tortuosity", "Tortuosity"),
            ("CoordinationNumber", "Coord. Number"),
            ("ConnectedPoreFraction", "Connected Pores"),
        ):
            if key in metadata:
                value = metadata[key]
                if key == "Permeability_mD":
                    value = f"{value} mD"
                computed.append((label, str(value)))

        self.computed_table.setRowCount(len(computed))
        for row, (metric, val) in enumerate(computed):
            self.computed_table.setItem(row, 0, QTableWidgetItem(metric))
            self.computed_table.setItem(row, 1, QTableWidgetItem(val))
        self._adjust_table_height(self.computed_table, max_height=400)
