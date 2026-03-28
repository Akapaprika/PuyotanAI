from PyQt6.QtCore import Qt
import puyotan_native as p
from .agents import HumanPlayerAgent


class GameplayController:
    """
    Translates hardware events (Qt keys, button signals) into ViewModel commands.
    Only routes input for players whose agent is HumanPlayerAgent.
    Has zero direct dependency on any widget; receives ViewModel by injection.
    """
    KEY_BINDINGS = {
        0: {
            "left":  Qt.Key.Key_Left,
            "right": Qt.Key.Key_Right,
            "rot_r": Qt.Key.Key_Up,
            "rot_l": Qt.Key.Key_Z,
            "drop":  Qt.Key.Key_Down,
        },
        1: {
            "left":  Qt.Key.Key_A,
            "right": Qt.Key.Key_D,
            "rot_r": Qt.Key.Key_W,
            "rot_l": Qt.Key.Key_Q,
            "drop":  Qt.Key.Key_S,
        },
    }

    def __init__(self, view_model):
        self.vm = view_model

    def _is_human(self, pid: int) -> bool:
        """Return True only if the given player slot is a HumanPlayerAgent."""
        return isinstance(self.vm.agents[pid], HumanPlayerAgent)

    # ------------------------------------------------------------------
    # Qt keyboard integration
    # ------------------------------------------------------------------
    def handle_key(self, key: Qt.Key) -> bool:
        """
        Route a Qt key press to the correct ViewModel command.
        Returns True if the key was consumed.
        Non-human players are silently skipped.
        """
        for pid, bindings in self.KEY_BINDINGS.items():
            if not self._is_human(pid):
                continue
            for action, bound_key in bindings.items():
                if key == bound_key:
                    self._dispatch(pid, action)
                    return True
        return False

    # ------------------------------------------------------------------
    # Button / UI signal integration
    # ------------------------------------------------------------------
    def handle_action(self, player_id: int, action_name: str) -> None:
        """Route a named action (from a button click) to the ViewModel.
        Button presses are silently dropped for non-human players.
        """
        if not self._is_human(player_id):
            return
        self._dispatch(player_id, action_name)

    # ------------------------------------------------------------------
    # Private dispatch table
    # ------------------------------------------------------------------
    def _dispatch(self, pid: int, action: str) -> None:
        dispatch = {
            "left":  lambda: self.vm.move_player(pid, -1),
            "right": lambda: self.vm.move_player(pid, 1),
            "rot_r": lambda: self.vm.rotate_player(pid, 1),
            "rot_l": lambda: self.vm.rotate_player(pid, -1),
            "drop":  lambda: self.vm.confirm_player(pid),
        }
        if action in dispatch:
            dispatch[action]()
