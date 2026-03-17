"""
gui/views/player_panel.py

A self-contained card widget representing one player's field + controls.
Emits the action_requested(player_id, action_name) signal when a button
is clicked, allowing the MainWindow to route it to the controller without
any direct coupling between the panel and the rest of the system.
"""
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt

from .board_widget import BoardWidget, BoardSnapshot


class PlayerPanel(QFrame):
    """
    Visual card for one player. Layout simplified for Phase 3:
    Frame counter is moved to the center of the window, and 
    status indicators are moved inside the BoardWidget info area.
    """
    action_requested = pyqtSignal(int, str)

    def __init__(self, player_id: int, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self.setObjectName("PlayerCard")

        # --- Root layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # --- Board ---
        self._board = BoardWidget()
        root.addWidget(self._board, stretch=1)

        # --- Controls ---
        root.addLayout(self._build_controls())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_snapshot(self, snapshot: BoardSnapshot) -> None:
        self._board.update_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_controls(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(6)

        buttons = [
            # Row 0: Navigation
            ("←",  "left",  0, 0),
            ("↓",  "drop",  0, 1),
            ("→",  "right", 0, 2),
            # Row 1: Rotation
            ("↺",  "rot_l", 1, 0),
            ("↻",  "rot_r", 1, 2),
        ]
        for spec in buttons:
            label, action = spec[0], spec[1]
            row, col = spec[2], spec[3]
            btn = QPushButton(label)
            btn.setObjectName("ActionBtn")
            # Width optimization: each button takes 1/3 of the panel area
            btn.setMinimumWidth(40)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("padding: 4px 0px; font-weight: bold; font-size: 14px;")
            btn.setToolTip(action)
            btn.clicked.connect(lambda _, a=action: self.action_requested.emit(self.player_id, a))
            grid.addWidget(btn, row, col)

        return grid
