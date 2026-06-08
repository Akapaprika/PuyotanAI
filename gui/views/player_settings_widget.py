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
    HumanPlayerAgent, AIPlayerAgent, EmptyPlayerAgent, BasePlayerAgent,
    BeamSearchAgent, HybridBeamOnnxAgent
)

# Resolve models directory relative to the project root (two levels up from this file)
# gui/views/player_settings_widget.py → gui/ → project root → models/
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


class PlayerSettingsWidget(QWidget):
    """
    Thin settings row: [P# ▼ Mode] [Browse...] [path label]
    """

    #: Emitted with (player_id, new_agent) whenever the agent type or model changes.
    agent_changed = pyqtSignal(int, object)

    _MODES = ["Human", "AI (ONNX)", "Beam Search", "Hybrid (Beam+ONNX)", "Empty (Solo)"]

    def __init__(self, player_id: int, allow_empty: bool = True, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self._model_path: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Label + Combo + Browse button
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        lbl = QLabel(f"P{player_id + 1}:")
        lbl.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 11px;")
        row1.addWidget(lbl)

        self._combo = QComboBox()
        modes = self._MODES if allow_empty else self._MODES[:2] + [self._MODES[2], self._MODES[3]]
        self._combo.addItems(modes)
        self._combo.setFixedWidth(130)
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._combo)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setFixedWidth(65)
        self._browse_btn.setStyleSheet("font-size: 11px;")
        self._browse_btn.setVisible(False)
        self._browse_btn.clicked.connect(self._on_browse)
        row1.addWidget(self._browse_btn)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: selected model path (hidden unless AI + path chosen)
        self._path_label = QLabel("")
        self._path_label.setVisible(False)
        self._path_label.setStyleSheet(
            "font-size: 10px; color: #64748b; padding-left: 28px;"
        )
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

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
        is_ai = current_mode in ("AI (ONNX)", "Hybrid (Beam+ONNX)")
        is_beam = current_mode in ("Beam Search", "Hybrid (Beam+ONNX)")

        self._browse_btn.setVisible(is_ai)
        self._beam_settings_widget.setVisible(is_beam)

        if not is_ai:
            self._path_label.setVisible(False)
            self._emit_agent()
        elif self._model_path:
            self._emit_agent()

    def _on_beam_param_changed(self, val: int) -> None:
        self._emit_agent()

    def _on_browse(self) -> None:
        start_dir = str(_DEFAULT_MODELS_DIR) if _DEFAULT_MODELS_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select ONNX model for P{self.player_id + 1}",
            start_dir,
            "ONNX Models (*.onnx)"
        )
        if path:
            self._model_path = path
            # Show truncated path (just filename + one parent dir)
            short = Path(path)
            display = f"…/{short.parent.name}/{short.name}" if short.parent.name else short.name
            self._path_label.setText(display)
            self._path_label.setToolTip(path)
            self._path_label.setVisible(True)
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
        if mode == "AI (ONNX)":
            if not self._model_path:
                return None, "Please click 'Browse' and select an ONNX model file."
            try:
                return AIPlayerAgent(self._model_path), None
            except Exception as e:
                return None, f"Failed to load AI model:\n{e}"
        if mode == "Beam Search":
            return BeamSearchAgent(beam_width=width, look_ahead=depth), None
        if mode == "Hybrid (Beam+ONNX)":
            if not self._model_path:
                return None, "Please click 'Browse' and select an ONNX model file."
            try:
                return HybridBeamOnnxAgent(self._model_path, beam_width=width, look_ahead=depth), None
            except Exception as e:
                return None, f"Failed to load Hybrid model:\n{e}"
        if mode == "Empty (Solo)":
            return EmptyPlayerAgent(), None
        return None, "Unknown mode."
