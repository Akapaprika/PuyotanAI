"""
gui/views/player_settings_widget.py

A compact widget row placed above each player's board (on the Setup screen).
Exposes a QComboBox for mode selection (Human / AI / Empty) and a
"Browse…" button that appears only when AI is selected.

Emits `agent_changed(player_id, BasePlayerAgent)` whenever the user
makes a new selection so the ViewModel can swap out the agent.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton,
    QFileDialog, QLabel, QSpinBox
)

from ..agents import (
    HumanPlayerAgent, EmptyPlayerAgent, BasePlayerAgent,
    BeamSearchAgent
)




class PlayerSettingsWidget(QWidget):
    """
    Thin settings row: [P# ▼ Mode] [Browse...] [path label]
    """

    #: Emitted with (player_id, new_agent) whenever the agent type or model changes.
    agent_changed = pyqtSignal(int, object)

    _MODES = ["Human", "Beam Search", "Empty (Solo)"]

    def __init__(self, player_id: int, allow_empty: bool = True, default_index: int = 0, parent=None):
        super().__init__(parent)
        self.player_id = player_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Label + Combo
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        lbl = QLabel(f"P{player_id + 1}:")
        lbl.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 11px;")
        row1.addWidget(lbl)

        self._combo = QComboBox()
        modes = self._MODES if allow_empty else self._MODES[:2]
        self._combo.addItems(modes)
        self._combo.setFixedWidth(130)
        if 0 <= default_index < len(modes):
            self._combo.setCurrentIndex(default_index)
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._combo)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 3: Beam Search settings (Width, Depth)
        self._beam_settings_widget = QWidget()
        beam_layout = QHBoxLayout(self._beam_settings_widget)
        beam_layout.setContentsMargins(28, 0, 0, 0)
        beam_layout.setSpacing(8)

        w_lbl = QLabel("Width:")
        w_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        beam_layout.addWidget(w_lbl)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(50, 2000)
        self._width_spin.setSingleStep(50)
        self._width_spin.setValue(500)
        self._width_spin.setFixedWidth(60)
        self._width_spin.setStyleSheet("font-size: 11px;")
        self._width_spin.valueChanged.connect(self._on_beam_param_changed)
        beam_layout.addWidget(self._width_spin)

        d_lbl = QLabel("Depth:")
        d_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        beam_layout.addWidget(d_lbl)

        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(2, 5)
        self._depth_spin.setValue(3)
        self._depth_spin.setFixedWidth(45)
        self._depth_spin.setStyleSheet("font-size: 11px;")
        self._depth_spin.valueChanged.connect(self._on_beam_param_changed)
        beam_layout.addWidget(self._depth_spin)

        beam_layout.addStretch()
        self._beam_settings_widget.setVisible(False)
        layout.addWidget(self._beam_settings_widget)

    # ------------------------------------------------------------------
    def _on_mode_changed(self, idx: int) -> None:
        current_mode = self._combo.currentText()
        is_beam = current_mode == "Beam Search"

        self._beam_settings_widget.setVisible(is_beam)
        self._emit_agent()

    def _on_beam_param_changed(self, val: int) -> None:
        self._emit_agent()



    def _emit_agent(self) -> None:
        agent, _ = self.get_agent_or_error()
        if agent is not None:
            self.agent_changed.emit(self.player_id, agent)

    def get_agent_or_error(self) -> tuple[BasePlayerAgent | None, str | None]:
        """Returns (agent, None) on success, or (None, error_message) on failure."""
        mode = self._combo.currentText()
        width = self._width_spin.value()
        depth = self._depth_spin.value()

        if mode == "Human":
            return HumanPlayerAgent(), None
        if mode == "Beam Search":
            return BeamSearchAgent(beam_width=width, look_ahead=depth), None
        if mode == "Empty (Solo)":
            return EmptyPlayerAgent(), None
        return None, "Unknown mode."
