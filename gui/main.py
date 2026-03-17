"""
gui/main.py

Application entry point.  Constructs the MVVM triad, loads the QSS theme,
and hands control to the Qt event loop.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# --- Auto-resolve native extension path ---
# This allows running without $env:PYTHONPATH manual set.
_PROJECT_ROOT = Path(__file__).parent.parent
_NATIVE_DIST = _PROJECT_ROOT / "native" / "dist"
if _NATIVE_DIST.exists() and str(_NATIVE_DIST) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIST))

# CRITICAL IMPORT ORDER: We MUST import the model (which imports the C++ puyotan_native extension)
# BEFORE importing PyQt6. Otherwise, on Windows, loading the PySide/PyQt DLLs first causes 
# an access violation (segfault) when the pybind11 extension tries to load.
from .model import GameModel
from .view_model import PuyotanViewModel

from PyQt6.QtWidgets import QApplication

from .controller import GameplayController
from .views import MainWindow


def _load_qss(relative_path: str) -> str:
    """Load a QSS stylesheet relative to this package directory."""
    qss_path = Path(__file__).parent / relative_path
    try:
        return qss_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Puyotan AI")
    app.setApplicationDisplayName("Puyotan AI — Match Viewer")

    # Apply global dark theme
    qss = _load_qss("assets/theme.qss")
    if qss:
        app.setStyleSheet(qss)

    # ── Wire MVVM ──────────────────────────────────────────────────────
    model   = GameModel(seed=42)
    vm      = PuyotanViewModel(model)
    ctrl    = GameplayController(vm)
    window  = MainWindow(vm, ctrl)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
