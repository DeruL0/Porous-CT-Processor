"""
Shared UI spacing and sizing constants for consistent panel layout.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QLayout, QSizePolicy


# Panel-level layout defaults
PANEL_MARGIN = (10, 10, 10, 10)
PANEL_SPACING = 8

# GroupBox internal layout defaults
GROUP_MARGIN = (8, 16, 8, 8)
GROUP_SPACING = 8

# Table behavior defaults
TABLE_MIN_HEIGHT = 150


def apply_panel_layout(layout: QLayout) -> None:
    layout.setContentsMargins(*PANEL_MARGIN)
    layout.setSpacing(PANEL_SPACING)


def apply_group_layout(layout: QLayout) -> None:
    layout.setContentsMargins(*GROUP_MARGIN)
    layout.setSpacing(GROUP_SPACING)


def make_description_label(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setMinimumWidth(100)
    return label


def set_primary_button_policy(button) -> None:
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
