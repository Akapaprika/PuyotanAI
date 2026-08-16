"""
gui/views/player_settings_widget.py

A clean, compact mode selection widget placed above each player's board (on the Setup screen).
All search parameters are managed automatically via the central `beam_config.json`.

Emits `agent_changed(player_id, BasePlayerAgent)` whenever the user
makes a new selection so the ViewModel can swap out the agent.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QComboBox, QLabel

from ai import AgentFactory, BasePlayerAgent


class PlayerSettingsWidget(QWidget):
    """
    Compact mode selector row: [P# ▼ Mode].
    """
    agent_changed = pyqtSignal(int, object)  # player_id, BasePlayerAgent

    def __init__(self, player_id: int, allow_empty: bool = True, default_index: int = 0, parent=None):
        super().__init__(parent)
        self.player_id = player_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(f"P{player_id + 1}:")
        lbl.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 10pt;")
        layout.addWidget(lbl)

        self._combo = QComboBox()
        modes = AgentFactory.get_modes(allow_empty=allow_empty)
        self._combo.addItems(modes)
        self._combo.setFixedWidth(180)
        self._combo.setFixedHeight(32)
        if 0 <= default_index < len(modes):
            self._combo.setCurrentIndex(default_index)
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._combo)

    # ------------------------------------------------------------------
    def _on_mode_changed(self, idx: int) -> None:
        agent, _ = self.get_agent_or_error()
        if agent is not None:
            self.agent_changed.emit(self.player_id, agent)

    def get_agent_or_error(self) -> tuple[BasePlayerAgent | None, str | None]:
        """
        Returns (agent, None) on success, or (None, error_message) on failure.
        The created agent automatically loads all latest search parameters from beam_config.json.
        """
        mode = self._combo.currentText()
        return AgentFactory.create_agent(mode)
