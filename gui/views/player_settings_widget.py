"""
gui/views/player_settings_widget.py

A compact widget row placed above each player's board (on the Setup screen).
Exposes a QComboBox for mode selection (Human / Beam Search / Empty) and parameters
for configuring beam search.

Emits `agent_changed(player_id, BasePlayerAgent)` whenever the user
makes a new selection so the ViewModel can swap out the agent.
"""
from __future__ import annotations

import puyotan_native as p
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton,
    QLabel, QSpinBox
)

from ..agents import (
    HumanPlayerAgent, EmptyPlayerAgent, BasePlayerAgent,
    BeamSearchAgent, _CONFIG_PATH
)


def _get_default_config() -> dict[str, int]:
    """Load default beam configuration from beam_config.json using the C++ bindings."""
    defaults = {"width": 15000, "depth": 25, "dbs": 6}
    try:
        cfg = p.load_solo_config(_CONFIG_PATH)
        defaults["width"] = cfg.beam_width
        defaults["depth"] = cfg.look_ahead
        defaults["dbs"] = cfg.dbs_max_similar
    except Exception:
        pass
    return defaults




class PlayerSettingsWidget(QWidget):
    """
    Thin settings row: [P# ▼ Mode] and optional Beam Search parameters.
    """

    #: Emitted with (player_id, new_agent) whenever the agent type or model changes.
    agent_changed = pyqtSignal(int, object)

    _MODES = ["Human", "Beam Search (Player)", "Beam Search (Enemy)", "Empty (Solo)"]

    def __init__(self, player_id: int, allow_empty: bool = True, default_index: int = 0, parent=None):
        super().__init__(parent)
        self.player_id = player_id

        defaults = _get_default_config()

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
        modes = self._MODES if allow_empty else self._MODES[:3]
        self._combo.addItems(modes)
        self._combo.setFixedWidth(130)
        if 0 <= default_index < len(modes):
            self._combo.setCurrentIndex(default_index)
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._combo)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 3: Beam Search settings (Width, Depth, DBS)
        self._beam_settings_widget = QWidget()
        beam_layout = QVBoxLayout(self._beam_settings_widget)
        beam_layout.setContentsMargins(28, 0, 0, 0)
        beam_layout.setSpacing(4)

        # Width row
        w_row = QHBoxLayout()
        w_row.setSpacing(4)
        w_row.setContentsMargins(0, 0, 0, 0)
        w_lbl = QLabel("Width:")
        w_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        w_lbl.setFixedWidth(40)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(50, 1000000)
        self._width_spin.setSingleStep(500)
        self._width_spin.setValue(defaults["width"])
        self._width_spin.setFixedWidth(75)
        self._width_spin.setStyleSheet("font-size: 11px;")
        self._width_spin.valueChanged.connect(self._on_beam_param_changed)
        w_row.addWidget(w_lbl)
        w_row.addWidget(self._width_spin)
        w_row.addStretch()
        beam_layout.addLayout(w_row)

        # Depth row
        d_row = QHBoxLayout()
        d_row.setSpacing(4)
        d_row.setContentsMargins(0, 0, 0, 0)
        d_lbl = QLabel("Depth:")
        d_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        d_lbl.setFixedWidth(40)
        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(2, 50)
        self._depth_spin.setValue(defaults["depth"])
        self._depth_spin.setFixedWidth(75)
        self._depth_spin.setStyleSheet("font-size: 11px;")
        self._depth_spin.valueChanged.connect(self._on_beam_param_changed)
        d_row.addWidget(d_lbl)
        d_row.addWidget(self._depth_spin)
        d_row.addStretch()
        beam_layout.addLayout(d_row)

        # DBS row
        dbs_row = QHBoxLayout()
        dbs_row.setSpacing(4)
        dbs_row.setContentsMargins(0, 0, 0, 0)
        dbs_lbl = QLabel("DBS:")
        dbs_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        dbs_lbl.setFixedWidth(40)
        self._dbs_spin = QSpinBox()
        self._dbs_spin.setRange(0, 100)
        self._dbs_spin.setSingleStep(1)
        self._dbs_spin.setValue(defaults["dbs"])
        self._dbs_spin.setFixedWidth(75)
        self._dbs_spin.setStyleSheet("font-size: 11px;")
        self._dbs_spin.valueChanged.connect(self._on_beam_param_changed)
        dbs_row.addWidget(dbs_lbl)
        dbs_row.addWidget(self._dbs_spin)
        dbs_row.addStretch()
        beam_layout.addLayout(dbs_row)

        self._beam_settings_widget.setVisible(False)
        layout.addWidget(self._beam_settings_widget)

    # ------------------------------------------------------------------
    def _on_mode_changed(self, idx: int) -> None:
        current_mode = self._combo.currentText()
        is_beam = "Beam Search" in current_mode

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
        dbs = self._dbs_spin.value()

        if mode == "Human":
            return HumanPlayerAgent(), None
        if mode == "Beam Search (Player)":
            return BeamSearchAgent(beam_width=width, look_ahead=depth, dbs_max_similar=dbs, is_enemy=False), None
        if mode == "Beam Search (Enemy)":
            return BeamSearchAgent(beam_width=width, look_ahead=depth, dbs_max_similar=dbs, is_enemy=True), None
        if mode == "Empty (Solo)":
            return EmptyPlayerAgent(), None
        return None, "Unknown mode."
