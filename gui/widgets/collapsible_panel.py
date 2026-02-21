"""
Simple collapsible container for dense sidebars.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QToolButton, QVBoxLayout, QWidget

from gui.ui_constants import GROUP_MARGIN, GROUP_SPACING


class CollapsiblePanel(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*GROUP_MARGIN)
        layout.setSpacing(GROUP_SPACING)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(bool(expanded))
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)
        layout.addWidget(content)

        self._content.setVisible(bool(expanded))

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(bool(checked))
