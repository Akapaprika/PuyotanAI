from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt

from .board_widget import BoardWidget, BoardSnapshot


class PlayerPanel(QFrame):
    """
    Visual card for one player.
    Shows board + action buttons. Buttons are invisible (but space-occupying)
    when the player is non-human to keep the layout symmetrical.
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

        self._buttons, self._controls_widget, self._controls_layout = self._build_controls()
        root.addWidget(self._controls_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_snapshot(self, snapshot: BoardSnapshot) -> None:
        self._board.update_snapshot(snapshot)

    def set_human_controlled(self, is_human: bool) -> None:
        """Show or hide each button individually while keeping the container
        always present (and thus the layout height/width stable)."""
        for btn in self._buttons:
            btn.setVisible(is_human)
            btn.setEnabled(is_human)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_controls(self):
        container = QFrame()
        grid = QGridLayout(container)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        # Fix the container height so it never collapses even when buttons
        # are hidden — measured against 2 rows of 32px buttons + spacing.
        container.setMinimumHeight(80)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        specs = [
            ("←",  "left",  0, 0),
            ("↓",  "drop",  0, 1),
            ("→",  "right", 0, 2),
            ("↺",  "rot_l", 1, 0),
            ("↻",  "rot_r", 1, 2),
        ]
        buttons = []
        for label, action, row, col in specs:
            btn = QPushButton(label)
            btn.setObjectName("ActionBtn")
            btn.setMinimumWidth(40)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("padding: 4px 0px; font-weight: bold; font-size: 14px;")
            btn.setToolTip(action)
            btn.clicked.connect(lambda _, a=action: self.action_requested.emit(self.player_id, a))
            grid.addWidget(btn, row, col)
            buttons.append(btn)

        return buttons, container, grid

