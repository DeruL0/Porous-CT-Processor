"""
Shared styles and CSS constants for the GUI components.
"""

# ==========================================
# Panel Title Styles
# ==========================================
PANEL_TITLE_STYLE = """
    font-size: 20px; 
    font-weight: bold; 
    margin-top: 15px; 
    margin-bottom: 5px;
    padding: 5px;
"""

# ==========================================
# Table Styles
# ==========================================
TABLE_STYLESHEET = """
    QTableWidget {
        background-color: transparent;
        gridline-color: #666666;
        font-size: 15px;
        border: 1px solid #555555;
    }
    QTableWidget::item {
        padding: 10px;
        border: 1px solid #444444;
    }
    QHeaderView::section {
        background-color: transparent;
        font-weight: bold;
        padding: 10px;
        border: 1px solid #555555;
        font-size: 15px;
    }
"""

# ==========================================
# Button Styles
# ==========================================
PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #4a90d9;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5a9fe9;
    }
    QPushButton:pressed {
        background-color: #3a80c9;
    }
"""

SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #555555;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #666666;
    }
"""

# ==========================================
# Slider Styles
# ==========================================
SLIDER_STYLESHEET = """
    QSlider::groove:horizontal {
        height: 8px;
        background: #444444;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #4a90d9;
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }
"""
