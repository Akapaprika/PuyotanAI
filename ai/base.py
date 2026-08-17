"""
ai/base.py

Abstract Base Classes and Mixins for AI Player Agents.
Provides thread-safe asynchronous search management reusable across GUI, Bot, and CLI.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import threading

import puyotan_native as p

# Sentinel returned by _AsyncSearchMixin._check_result() while a search thread is alive.
_STILL_THINKING = object()


class _AsyncSearchMixin:
    """
    Reusable thread-management helpers for AI agents that run search in a background
    thread. Call _init_async() from __init__ before using any other method.
    """

    def _init_async(self) -> None:
        self._thread: threading.Thread | None = None
        self._result: Any = None
        self._lock = threading.Lock()

    def _check_result(self) -> Any:
        """
        Non-blocking poll of the background search.

        Returns
        -------
        result          — the stored value if search finished (clears state).
        _STILL_THINKING — a search thread is alive but not yet done.
        None            — no thread is running; caller should start one.
        """
        with self._lock:
            if self._result is not None:
                r, self._result, self._thread = self._result, None, None
                return r
            if self._thread is not None and self._thread.is_alive():
                return _STILL_THINKING
        return None

    def _launch(self, target) -> None:
        """Start a new daemon search thread."""
        t = threading.Thread(target=target, daemon=True)
        with self._lock:
            self._thread = t
        t.start()

    def _store_result(self, result: Any) -> None:
        """Called from inside the worker thread to publish the result."""
        with self._lock:
            self._result = result

    def reset_search(self) -> None:
        """Cancel/clear any current search state."""
        with self._lock:
            self._result = None
            self._thread = None


class BasePlayerAgent(ABC):
    """Abstract interface for a player controller."""

    # Subclasses override these as class-level flags to avoid isinstance checks.
    is_human: bool = False
    is_empty: bool = False

    @abstractmethod
    def get_action(self, match: Any, player_id: int, pres: Any = None) -> p.Action | None:
        """
        Return an action for the current decision point, or None if the
        agent is still waiting (e.g. thinking in thread, or human hasn't pressed confirm).

        Parameters
        ----------
        match      : GameModel or GameState — wrapper providing get_player_state(), get_tsumo()
        player_id  : int — 0 or 1
        pres       : Optional presentation state (used by HumanPlayerAgent in GUI)
        """

    def reset(self) -> None:
        """
        Called when a match starts or restarts to reset internal search session and reload config.
        Default is a no-op; override in subclasses.
        """
