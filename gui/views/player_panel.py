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
    Visual card for one player.
    Shows board + action buttons. Buttons are hidden when the player is non-human
    to prevent accidental input reaching the ViewModel.
    """
    action_requested = pyqtSignal(int, str)

    def __init__(self, player_id: int, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self.setObjectName("PlayerCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        self._board = BoardWidget()
        root.addWidget(self._board, stretch=1)

        self._controls_widget, self._controls_layout = self._build_controls()
        root.addWidget(self._controls_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_snapshot(self, snapshot: BoardSnapshot) -> None:
        self._board.update_snapshot(snapshot)

    def set_human_controlled(self, is_human: bool) -> None:
        """Show/hide action buttons based on whether this player is human."""
        self._controls_widget.setVisible(is_human)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_controls(self):
        container = QFrame()
        grid = QGridLayout(container)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        buttons = [
            ("←",  "left",  0, 0),
            ("↓",  "drop",  0, 1),
            ("→",  "right", 0, 2),
            ("↺",  "rot_l", 1, 0),
            ("↻",  "rot_r", 1, 2),
        ]
        for spec in buttons:
            label, action = spec[0], spec[1]
            row, col = spec[2], spec[3]
            btn = QPushButton(label)
            btn.setObjectName("ActionBtn")
            btn.setMinimumWidth(40)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("padding: 4px 0px; font-weight: bold; font-size: 14px;")
            btn.setToolTip(action)
            btn.clicked.connect(lambda _, a=action: self.action_requested.emit(self.player_id, a))
            grid.addWidget(btn, row, col)

        return container, grid

