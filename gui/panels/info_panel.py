"""
Info Panel components for displaying sample information and statistics.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, 
    QTableWidget, QTableWidgetItem, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.styles import PANEL_TITLE_STYLE, TABLE_STYLESHEET


class StatisticsPanel(QWidget):
    """
    Panel to display advanced statistics and visualizations for PNM analysis.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("📈 Statistics & Analysis")
        title.setStyleSheet(PANEL_TITLE_STYLE)
        layout.addWidget(title)
        
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(280) # Ensure charts are visible
        
        # 1. Pore Size Chart
        self.fig_pore = Figure(figsize=(5, 3), dpi=80)
        self.canvas_pore = FigureCanvas(self.fig_pore)
        self.tabs.addTab(self.canvas_pore, "Pore Sizes")
        
        # 2. Throat Size Chart
        self.fig_throat = Figure(figsize=(5, 3), dpi=80)
        self.canvas_throat = FigureCanvas(self.fig_throat)
        self.tabs.addTab(self.canvas_throat, "Throat Sizes")
        
        layout.addWidget(self.tabs)


        # Statistics Table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setStyleSheet(TABLE_STYLESHEET)
        
        layout.addWidget(self.stats_table)
        
        # UI behavior - auto growth
        self.stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addStretch()




    def update_statistics(self, metadata):
        """Update the panel with new statistics from VolumeData metadata."""
        self.fig_pore.clear()
        self.fig_throat.clear()
        self.stats_table.setRowCount(0)

        pore_dist = metadata.get("PoreSizeDistribution", {})
        throat_dist = metadata.get("ThroatSizeDistribution", {})
        largest_pore = metadata.get("LargestPoreRatio", "N/A")
        throat_stats = metadata.get("ThroatStats", {})
        pore_count = metadata.get("PoreCount", 0)
        connection_count = metadata.get("ConnectionCount", 0)

        # Plot Pore histogram
        if pore_dist.get("bins") and pore_dist.get("counts"):
            ax = self.fig_pore.add_subplot(111)
            bins = np.array(pore_dist["bins"])
            counts = pore_dist["counts"]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, counts, width=np.diff(bins),
                   alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Pore Radius (mm)')
            ax.set_ylabel('Count')
            ax.set_title('Pore Size Distribution')
            ax.grid(True, alpha=0.3)
            self.fig_pore.tight_layout()
            
        # Plot Throat histogram
        if throat_dist.get("bins") and throat_dist.get("counts"):
            ax = self.fig_throat.add_subplot(111)
            bins = np.array(throat_dist["bins"])
            counts = throat_dist["counts"]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, counts, width=np.diff(bins),
                   alpha=0.7, color='indianred', edgecolor='black')
            ax.set_xlabel('Throat Radius (mm)')
            ax.set_ylabel('Count')
            ax.set_title('Throat Size Distribution')
            ax.grid(True, alpha=0.3)
            self.fig_throat.tight_layout()

        self.canvas_pore.draw()
        self.canvas_throat.draw()

        # Stats table
        stats_data = [
            ("🔴 Total Pores", str(pore_count)),
            ("🔗 Connections", str(connection_count)),
            ("👑 Largest Pore %", str(largest_pore)),
        ]

        if throat_stats:
            stats_data.extend([
                ("📉 Throat Min", f"{throat_stats.get('min', 0):.3f} mm"),
                ("📈 Throat Max", f"{throat_stats.get('max', 0):.3f} mm"),
                ("⚖️ Throat Mean", f"{throat_stats.get('mean', 0):.3f} mm"),
            ])

        self.stats_table.setRowCount(len(stats_data))
        for row, (metric, value) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))
            
        self._adjust_table_height(self.stats_table, max_height=350)


    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400):
        """Dynamically adjust table height to fit contents."""
        table.doItemsLayout() # Ensure rows are laid out
        
        # Calculate total height: header + all rows + small margin
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        
        height += 5 # Add safe margin
        
        # Clamp between reasonable min/max
        clamped_height = max(100, min(height, max_height))
        table.setFixedHeight(clamped_height)
        
        # Toggle scrollbar
        if height > max_height:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def clear(self):
        """Clear all statistics."""
        self.figure.clear()
        self.canvas.draw()
        self.stats_table.setRowCount(0)


class InfoPanel(QGroupBox):
    """Enhanced panel to display sample metadata and computed statistics."""

    def __init__(self, title: str = "📊 Sample Information"):
        super().__init__()
        self.custom_title = title
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        title_lbl = QLabel(self.custom_title)
        title_lbl.setStyleSheet(PANEL_TITLE_STYLE)
        main_layout.addWidget(title_lbl)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Info table
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.info_table.setStyleSheet(TABLE_STYLESHEET)
        self.info_table.setShowGrid(True)
        splitter.addWidget(self.info_table)
        
        # Computed Metrics section
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.computed_label = QLabel("<b>🧮 Computed Metrics</b>")
        self.computed_label.setStyleSheet(PANEL_TITLE_STYLE)
        bottom_layout.addWidget(self.computed_label)
        
        self.computed_table = QTableWidget()
        self.computed_table.setColumnCount(2)
        self.computed_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.computed_table.horizontalHeader().setStretchLastSection(True)
        self.computed_table.verticalHeader().setVisible(False)
        self.computed_table.setAlternatingRowColors(True)
        self.computed_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.computed_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.computed_table.setStyleSheet(TABLE_STYLESHEET)
        self.computed_table.setShowGrid(True)
        bottom_layout.addWidget(self.computed_table)

        main_layout.addWidget(self.info_table)
        main_layout.addWidget(self.computed_label)
        main_layout.addWidget(self.computed_table)
        
        # Disable internal scrollbars
        self.info_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.computed_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.setLayout(main_layout)

    def _adjust_table_height(self, table: QTableWidget, max_height: int = 400):
        """Dynamically adjust table height to fit contents."""
        table.doItemsLayout()
        height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            height += table.rowHeight(i)
        height += 5
        
        clamped_height = max(60, min(height, max_height))
        table.setFixedHeight(clamped_height)
        
        if height > max_height:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


    def update_info(self, type_str: str, dim: tuple, spacing: tuple, metadata: dict):
        """Update with new data including computed scientific metrics."""
        self.info_table.setRowCount(0)
        self.computed_table.setRowCount(0)

        # Basic Info
        basic_data = [("📁 Data Type", type_str)]

        if dim != (0, 0, 0):
            basic_data.append(("📐 Dimensions", f"{dim[0]} × {dim[1]} × {dim[2]}"))
        elif 'MeshPoints' in metadata:
            basic_data.append(("🔷 Mesh Points", str(metadata['MeshPoints'])))

        basic_data.append(("📏 Voxel Size", f"{spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f} mm"))

        priority_keys = ['Porosity', 'PoreCount', 'ConnectionCount']
        icons = {'Porosity': '💨', 'PoreCount': '🔴', 'ConnectionCount': '🔗'}
        for key in priority_keys:
            if key in metadata:
                icon = icons.get(key, '•')
                val = metadata[key]
                if isinstance(val, float):
                    val = f"{val:.2f}%"
                basic_data.append((f"{icon} {key}", str(val)))

        self.info_table.setRowCount(len(basic_data))
        for row, (prop, val) in enumerate(basic_data):
            self.info_table.setItem(row, 0, QTableWidgetItem(prop))
            self.info_table.setItem(row, 1, QTableWidgetItem(val))
            
        self._adjust_table_height(self.info_table, max_height=300)



        # Computed Metrics
        computed = []

        if dim != (0, 0, 0):
            total_voxels = dim[0] * dim[1] * dim[2]
            computed.append(("🔢 Total Voxels", f"{total_voxels:,}"))

            phys_volume = dim[0] * spacing[0] * dim[1] * spacing[1] * dim[2] * spacing[2]
            if phys_volume > 1000:
                computed.append(("🧊 Physical Volume", f"{phys_volume / 1000:.2f} cm³"))
            else:
                computed.append(("🧊 Physical Volume", f"{phys_volume:.2f} mm³"))

            phys_dim = (dim[0] * spacing[0], dim[1] * spacing[1], dim[2] * spacing[2])
            computed.append(("📏 Physical Size", f"{phys_dim[0]:.1f} × {phys_dim[1]:.1f} × {phys_dim[2]:.1f} mm"))

            mem_mb = (total_voxels * 4) / (1024 * 1024)
            computed.append(("💾 Memory (Est.)", f"{mem_mb:.1f} MB"))

        if 'ThroatStats' in metadata:
            ts = metadata['ThroatStats']
            if 'mean' in ts:
                computed.append(("⭕ Avg Throat Radius", f"{ts['mean']:.3f} mm"))
            if 'min' in ts and 'max' in ts:
                computed.append(("📶 Throat Range", f"{ts['min']:.3f} - {ts['max']:.3f} mm"))

        if 'LargestPoreRatio' in metadata:
            computed.append(("👑 Largest Pore %", str(metadata['LargestPoreRatio'])))
        
        if 'Permeability_mD' in metadata:
            computed.append(("🔬 Permeability", f"{metadata['Permeability_mD']} mD"))
        if 'Tortuosity' in metadata:
            computed.append(("🌀 Tortuosity", str(metadata['Tortuosity'])))
        if 'CoordinationNumber' in metadata:
            computed.append(("🔗 Coord. Number", str(metadata['CoordinationNumber'])))
        if 'ConnectedPoreFraction' in metadata:
            computed.append(("📡 Connected Pores", str(metadata['ConnectedPoreFraction'])))

        self.computed_table.setRowCount(len(computed))
        for row, (metric, val) in enumerate(computed):
            self.computed_table.setItem(row, 0, QTableWidgetItem(metric))
            self.computed_table.setItem(row, 1, QTableWidgetItem(val))
            
        self._adjust_table_height(self.computed_table, max_height=400)


