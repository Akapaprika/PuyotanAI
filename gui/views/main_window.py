"""
gui/views/main_window.py

QMainWindow that composes:
  - Page 0 (Setup): SetupWidget — pick agents, press Start
  - Page 1 (Game):  Status bar + two PlayerPanels + frame counter

Uses QStackedWidget to switch between them.
Restart / Game Over always returns to the Setup page.
"""
from __future__ import annotations

import puyotan_native as p

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QMessageBox, QLabel, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent, QColor

from .status_bar_widget import StatusBarWidget
from .player_panel import PlayerPanel
from .setup_widget import SetupWidget
from .board_widget import (
    BoardSnapshot, PieceSpec, piece_spec_from_native, CELL_COLORS
)
from ..agents import HumanPlayerAgent
from .. import config


def _state_label(pres) -> str:
    if pres.rigid_frames > 0:
        return "LOCKED"
    if pres.has_decision:
        return "READY" if pres.confirmed else "THINKING"
    return "WAITING"


def _build_snapshot(vm, pid: int) -> BoardSnapshot:
    pres  = vm.players[pid]
    field = vm.get_player_field(pid)
    next1 = piece_spec_from_native(vm.get_next_piece(pid, 1), CELL_COLORS)
    next2 = piece_spec_from_native(vm.get_next_piece(pid, 2), CELL_COLORS)

    ghost_axis = QColor(*pres.ghost_color_axis)
    ghost_sub  = QColor(*pres.ghost_color_sub)

    chain_current, chain_last = vm.get_chain_count(pid)

    return BoardSnapshot(
        field=field,
        next1=next1,
        next2=next2,
        ghost_x=pres.x,
        ghost_rot=pres.rotation,
        ghost_axis_col=ghost_axis,
        ghost_sub_col=ghost_sub,
        show_ghost=pres.has_decision,
        score=vm.get_player_score(pid),
        non_active_ojama=vm.get_player_ojama(pid)[0],
        active_ojama=vm.get_player_ojama(pid)[1],
        chain_count=chain_current,
        last_chain=chain_last,
        state=_state_label(pres)
    )


class MainWindow(QMainWindow):
    """Root application window with Setup / Game pages."""

    _PAGE_SETUP = 0
    _PAGE_GAME  = 1

    def __init__(self, view_model, controller, parent=None):
        super().__init__(parent)
        self.vm   = view_model
        self.ctrl = controller

        self.setWindowTitle("Puyotan AI — Match Viewer")
        self.setMinimumSize(780, 580)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ── Stacked pages ───────────────────────────────────────────────
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0: Setup
        self._setup_page = SetupWidget()
        self._setup_page.start_requested.connect(self._on_start)
        self._stack.addWidget(self._setup_page)

        # Page 1: Game
        self._stack.addWidget(self._build_game_page())

        # Show setup first
        self._stack.setCurrentIndex(self._PAGE_SETUP)

        # ── Wire ViewModel signals ──────────────────────────────────────
        self.vm.state_changed.connect(self._refresh)
        self.vm.game_over.connect(self._on_game_over)

        # ── QTimer for game loop (starts only when game begins) ─────────
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.vm.update)

    # ------------------------------------------------------------------
    # Build game page
    # ------------------------------------------------------------------
    def _build_game_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # Status bar
        self._status_bar = StatusBarWidget(
            initial_interval_ms=config.VIRTUAL_FRAME_INTERVAL_MS,
            initial_seed=getattr(config, 'RANDOM_SEED', 42)
        )
        root.addWidget(self._status_bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d3748; margin: 0;")
        root.addWidget(sep)

        # Player panels
        panels_row = QHBoxLayout()
        panels_row.setSpacing(10)

        self._panels: list[PlayerPanel] = []
        p1 = PlayerPanel(0)
        p1.action_requested.connect(self._on_player_action)
        panels_row.addWidget(p1, stretch=5)
        self._panels.append(p1)

        # Center frame counter
        center_col = QVBoxLayout()
        center_col.addStretch()
        self._frame_counter = QLabel("0")
        self._frame_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_counter.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #6366f1;"
        )
        f_lbl = QLabel("FRAME")
        f_lbl.setStyleSheet("font-size: 10px; color: #94a3b8;")
        f_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_col.addWidget(f_lbl)
        center_col.addWidget(self._frame_counter)
        center_col.addStretch()
        panels_row.addLayout(center_col, stretch=1)

        p2 = PlayerPanel(1)
        p2.action_requested.connect(self._on_player_action)
        panels_row.addWidget(p2, stretch=5)
        self._panels.append(p2)

        root.addLayout(panels_row, stretch=1)

        # Wire status bar
        self._status_bar.restart_requested.connect(self._on_restart)
        self._status_bar.interval_changed.connect(self._on_interval_changed)
        self._status_bar.seed_changed.connect(self._on_seed_changed)

        return page

    # ------------------------------------------------------------------
    # Keyboard routing
    # ------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._stack.currentIndex() == self._PAGE_GAME:
            if not self.ctrl.handle_key(Qt.Key(event.key())):
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_start(self, agents: list) -> None:
        """Commit agent choices, configure panels, and begin the match."""
        for pid, agent in enumerate(agents):
            self.vm.set_agent(pid, agent)
            is_human = isinstance(agent, HumanPlayerAgent)
            self._panels[pid].set_human_controlled(is_human)

        self.vm.restart()
        self._stack.setCurrentIndex(self._PAGE_GAME)
        self._timer.start()
        self._refresh()

    def _on_player_action(self, player_id: int, action: str) -> None:
        self.ctrl.handle_action(player_id, action)

    def _on_interval_changed(self, ms: int) -> None:
        config.VIRTUAL_FRAME_INTERVAL_MS = ms

    def _on_seed_changed(self, seed: int) -> None:
        self.vm.model.seed = seed

    def _on_game_over(self, status_text: str) -> None:
        self._timer.stop()
        QMessageBox.information(self, "Game Over", f"Match ended!\n\n{status_text}")
        self._stack.setCurrentIndex(self._PAGE_SETUP)

    def _on_restart(self) -> None:
        """Return to setup screen instead of immediately restarting."""
        self._timer.stop()
        self._stack.setCurrentIndex(self._PAGE_SETUP)

    def _refresh(self) -> None:
        if self._stack.currentIndex() != self._PAGE_GAME:
            return
        self._frame_counter.setText(str(self.vm.frame))
        self._status_bar.set_status(self.vm.status_text)
        for pid, panel in enumerate(self._panels):
            snap = _build_snapshot(self.vm, pid)
            panel.update_snapshot(snap)
