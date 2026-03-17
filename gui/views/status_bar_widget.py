"""
gui/views/status_bar_widget.py

Top-of-window information / control bar.
Emits signals instead of calling objects directly, keeping it decoupled.
"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSpinBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt


class StatusBarWidget(QWidget):
    """
    Displays match-level information and provides global controls.

    Signals
    -------
    restart_requested()
        Emitted when the Restart button is clicked.
    interval_changed(ms: int)
        Emitted when the user changes the frame-interval spinbox.
    """
    restart_requested  = pyqtSignal()
    interval_changed   = pyqtSignal(int)
    seed_changed       = pyqtSignal(int)

    def __init__(self, initial_interval_ms: int = 100, initial_seed: int = 1, parent=None):
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 4, 10, 4)
        root.setSpacing(12)

        # ── Left: match info ────────────────────────────────────────────
        info_col = QVBoxLayout()
        info_col.setSpacing(1)
        self._status_label = QLabel("Status: —")
        self._status_label.setStyleSheet("font-size: 13px; color: #94a3b8; font-weight: bold;")
        info_col.addWidget(self._status_label)
        root.addLayout(info_col)

        root.addStretch()

        # ── Seed control ──────────────────────────────────────────────
        seed_col = QHBoxLayout()
        seed_col.setSpacing(4)
        seed_lbl = QLabel("Seed:")
        seed_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        seed_col.addWidget(seed_lbl)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(initial_seed)
        self._seed_spin.setStyleSheet("font-size: 11px;")
        self._seed_spin.valueChanged.connect(self.seed_changed.emit)
        seed_col.addWidget(self._seed_spin)
        root.addLayout(seed_col)

        # Small gap
        root.addSpacing(8)

        # ── Centre: speed control ────────────────────────────────────────
        speed_col = QHBoxLayout()
        speed_col.setSpacing(4)
        lbl = QLabel("Step Interval:")
        lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        speed_col.addWidget(lbl)
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(100, 9999)
        self._interval_spin.setSingleStep(100)
        self._interval_spin.setValue(initial_interval_ms)
        self._interval_spin.setSuffix(" ms")
        self._interval_spin.setToolTip("Time between virtual frames (lower = faster)")
        self._interval_spin.setStyleSheet("font-size: 11px;")
        self._interval_spin.valueChanged.connect(self.interval_changed.emit)
        speed_col.addWidget(self._interval_spin)
        root.addLayout(speed_col)

        # Divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #2d3748;")
        root.addWidget(sep)

        # ── Right: restart button ────────────────────────────────────────
        self._restart_btn = QPushButton("⟳ RESTART")
        self._restart_btn.setStyleSheet("font-size: 11px; font-weight: bold; padding: 4px 10px;")
        self._restart_btn.setToolTip("Restart the match from scratch")
        self._restart_btn.clicked.connect(self.restart_requested.emit)
        root.addWidget(self._restart_btn)

    def set_status(self, text: str) -> None:
        self._status_label.setText(f"Status: {text}")
