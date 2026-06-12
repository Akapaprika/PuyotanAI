"""
gui/views/setup_widget.py

Pre-match setup screen.
Shows player mode selectors for both players, a seed value setting, and a "Start" button.
Emits `start_requested(agents: list[BasePlayerAgent], seed: int)` when the user commits.
"""
from __future__ import annotations

import random
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QDoubleSpinBox, QMessageBox
)

from .player_settings_widget import PlayerSettingsWidget
from ..agents import BasePlayerAgent, HumanPlayerAgent


class SetupWidget(QWidget):
    """
    Full-screen setup panel.
    Both players choose Human / AI / Empty here before the match starts.
    Emits start_requested with the confirmed list of agents and the seed.
    """
    start_requested = pyqtSignal(list, int)   # list[BasePlayerAgent], seed

    def __init__(self, parent=None):
        super().__init__(parent)

        self._agents: list[BasePlayerAgent | None] = [HumanPlayerAgent(), HumanPlayerAgent()]

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(20)

        # Title
        title = QLabel("Puyotan AI — Match Setup")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #6366f1; margin-bottom: 8px;"
        )
        root.addWidget(title)

        # Divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d3748;")
        root.addWidget(sep)

        # Player settings row (side by side)
        settings_row = QHBoxLayout()
        settings_row.setSpacing(40)

        self._p1_settings = PlayerSettingsWidget(0, allow_empty=False)
        self._p1_settings.agent_changed.connect(lambda pid, a: self._on_agent_changed(pid, a))
        settings_row.addWidget(self._p1_settings)

        vs_lbl = QLabel("VS")
        vs_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vs_lbl.setStyleSheet("font-size: 28px; font-weight: bold; color: #475569;")
        settings_row.addWidget(vs_lbl)

        self._p2_settings = PlayerSettingsWidget(1, allow_empty=True)
        self._p2_settings.agent_changed.connect(lambda pid, a: self._on_agent_changed(pid, a))
        settings_row.addWidget(self._p2_settings)

        root.addLayout(settings_row)
        root.addStretch()

        # Seed input row
        seed_row = QHBoxLayout()
        seed_row.setSpacing(10)
        seed_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        seed_lbl = QLabel("Match Seed:")
        seed_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #94a3b8;")
        seed_row.addWidget(seed_lbl)

        # Use QDoubleSpinBox with 0 decimals to support full 32-bit unsigned integers (up to 4294967295)
        self._seed_spin = QDoubleSpinBox()
        self._seed_spin.setDecimals(0)
        self._seed_spin.setRange(1, 4294967295)
        self._seed_spin.setValue(1)
        self._seed_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._seed_spin.setFixedWidth(150)
        self._seed_spin.setFixedHeight(30)
        self._seed_spin.setStyleSheet(
            "QDoubleSpinBox {"
            "  font-size: 14px; font-weight: bold; color: white;"
            "  background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 2px 5px;"
            "}"
        )
        seed_row.addWidget(self._seed_spin)

        # Randomize Seed button
        self._rand_btn = QPushButton("Random")
        self._rand_btn.setFixedWidth(80)
        self._rand_btn.setFixedHeight(30)
        self._rand_btn.setStyleSheet(
            "QPushButton {"
            "  background: #334155; color: white; font-size: 12px; font-weight: bold; border-radius: 6px;"
            "}"
            "QPushButton:hover { background: #475569; }"
            "QPushButton:pressed { background: #1e293b; }"
        )
        self._rand_btn.clicked.connect(self._on_random_seed)
        seed_row.addWidget(self._rand_btn)

        root.addLayout(seed_row)
        root.addSpacing(10)

        # Start button
        self._start_btn = QPushButton("▶  Start Match")
        self._start_btn.setFixedHeight(50)
        self._start_btn.setStyleSheet(
            "QPushButton {"
            "  background: #6366f1; color: white; font-size: 16px;"
            "  font-weight: bold; border-radius: 8px; border: none;"
            "}"
            "QPushButton:hover { background: #4f46e5; }"
            "QPushButton:pressed { background: #4338ca; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        root.addWidget(self._start_btn)

    # ------------------------------------------------------------------
    def _on_agent_changed(self, pid: int, agent: BasePlayerAgent) -> None:
        self._agents[pid] = agent

    def _on_random_seed(self) -> None:
        r = random.randint(1, 4294967295)
        self._seed_spin.setValue(float(r))

    def _on_start(self) -> None:
        # Validate P1
        a1, err1 = self._p1_settings.get_agent_or_error()
        if err1:
            QMessageBox.critical(self, "Player 1 Setup Error", f"Cannot start match.\nP1 error: {err1}")
            return

        # Validate P2
        a2, err2 = self._p2_settings.get_agent_or_error()
        if err2:
            QMessageBox.critical(self, "Player 2 Setup Error", f"Cannot start match.\nP2 error: {err2}")
            return

        seed_val = int(self._seed_spin.value())
        self.start_requested.emit([a1, a2], seed_val)
