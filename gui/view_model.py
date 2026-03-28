import puyotan_native as p
from PyQt6.QtCore import QObject, pyqtSignal, QElapsedTimer
from . import config
from .agents import BasePlayerAgent, HumanPlayerAgent

class PlayerPresentationState:
    """State for a single player as seen by the UI."""
    def __init__(self):
        self.x = 2
        self.rotation = p.Rotation.Up
        self.confirmed = False
        self.rigid_frames = 0
        self.has_decision = False
        # Pre-blended colors for UI convenience
        self.ghost_color_axis = (255, 255, 255)
        self.ghost_color_sub = (255, 255, 255)

class PuyotanViewModel(QObject):
    """
    The ViewModel for the Puyotan GUI.
    Encapsulates game logic, agent dispatching, and timing.
    Emits signals when the view should be updated.
    """
    state_changed = pyqtSignal()
    game_over = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.players = [PlayerPresentationState(), PlayerPresentationState()]
        # Default: both players are human
        self.agents: list[BasePlayerAgent] = [HumanPlayerAgent(), HumanPlayerAgent()]
        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_step_time = self.timer.elapsed()
        
        self.p_colors = {
            p.Cell.Red: config.COLORS["Red"],
            p.Cell.Green: config.COLORS["Green"],
            p.Cell.Blue: config.COLORS["Blue"],
            p.Cell.Yellow: config.COLORS["Yellow"],
            p.Cell.Ojama: config.COLORS["Ojama"],
            p.Cell.Empty: config.COLORS["Empty"]
        }
        self.update_presentation()

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------
    def set_agent(self, player_id: int, agent: BasePlayerAgent) -> None:
        """Hot-swap the agent for a player (can be called at any time)."""
        self.agents[player_id] = agent
        self.reset_player_input(player_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def update(self):
        """Main update loop for logic and timing (called by QTimer)."""
        if not self.model.is_playing():
            return

        current_time = self.timer.elapsed()
        state_was_changed = False

        mask = self.model.get_decision_mask()

        if mask == 0:
            # All players are in auto-frames (chain, ojama, etc.) — just tick.
            if current_time - self.last_step_time >= config.VIRTUAL_FRAME_INTERVAL_MS:
                if self.model.step():
                    self.last_step_time = current_time
                    state_was_changed = True
                    new_mask = self.model.get_decision_mask()
                    for pid in [0, 1]:
                        if new_mask & (1 << pid):
                            self.reset_player_input(pid)
        else:
            # Ask each agent if it has an action ready.
            for pid in [0, 1]:
                if not (mask & (1 << pid)):
                    continue
                action = self.agents[pid].get_action(self.model, pid, self.players[pid])
                if action is not None:
                    if self.model.set_action(pid, action):
                        state_was_changed = True

        if state_was_changed:
            self.update_presentation()
            if not self.model.is_playing():
                self.game_over.emit(self.model.get_status_text())

    def update_presentation(self):
        """Refresh presentation-only state from the model."""
        mask = self.model.get_decision_mask()
        for pid in [0, 1]:
            player_state = self.model.get_player_state(pid)
            pres = self.players[pid]

            pres.rigid_frames = player_state.current_action.remaining_frame
            pres.has_decision = bool(mask & (1 << pid))

            if not pres.has_decision:
                pres.confirmed = False

            piece = self.model.get_piece(pid, 0)
            pres.ghost_color_axis = self._blend_ghost(self.p_colors.get(piece.axis, (255,255,255)))
            pres.ghost_color_sub = self._blend_ghost(self.p_colors.get(piece.sub, (255,255,255)))
            
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Human input helpers (called by controller — ignored for AI/Empty)
    # ------------------------------------------------------------------
    def move_player(self, pid, dx):
        if not self.model.is_playing():
            return
        # Guard against stale has_decision — always verify from engine
        if not (self.model.get_decision_mask() & (1 << pid)):
            return
        if not self.players[pid].confirmed:
            rot = self.players[pid].rotation
            min_x = 0
            max_x = 5
            if rot == p.Rotation.Left: min_x = 1
            if rot == p.Rotation.Right: max_x = 4
            new_x = max(min_x, min(max_x, self.players[pid].x + dx))
            if new_x != self.players[pid].x:
                self.players[pid].x = new_x
                self.state_changed.emit()

    def rotate_player(self, pid, direction):
        if not self.model.is_playing():
            return
        # Guard against stale has_decision — always verify from engine
        if not (self.model.get_decision_mask() & (1 << pid)):
            return
        if not self.players[pid].confirmed:
            rot_order = [p.Rotation.Up, p.Rotation.Right, p.Rotation.Down, p.Rotation.Left]
            idx = rot_order.index(self.players[pid].rotation)
            new_rot = rot_order[(idx + direction) % 4]
            self.players[pid].rotation = new_rot
            if new_rot == p.Rotation.Left and self.players[pid].x == 0:
                self.players[pid].x = 1
            elif new_rot == p.Rotation.Right and self.players[pid].x == 5:
                self.players[pid].x = 4
            self.state_changed.emit()

    def confirm_player(self, pid):
        if not self.model.is_playing():
            return
        # Guard against stale has_decision — always verify from engine
        if not (self.model.get_decision_mask() & (1 << pid)):
            return
        if not self.players[pid].confirmed:
            self.players[pid].confirmed = True
            action = p.Action(p.ActionType.PUT, self.players[pid].x, self.players[pid].rotation)
            self.model.set_action(pid, action)
            self.update_presentation()

    def reset_player_input(self, pid):
        self.players[pid].x = 2
        self.players[pid].rotation = p.Rotation.Up
        self.players[pid].confirmed = False
        self.players[pid].has_decision = True

    def restart(self):
        self.model.restart()
        for pid in [0, 1]:
            self.reset_player_input(pid)
        self.last_step_time = self.timer.elapsed()
        self.update_presentation()

    def _blend_ghost(self, color):
        bg = config.COLORS["Background"]
        alpha = config.COLORS["GhostAlpha"] / 255.0
        return (
            int(color[0]*alpha + bg[0]*(1-alpha)),
            int(color[1]*alpha + bg[1]*(1-alpha)),
            int(color[2]*alpha + bg[2]*(1-alpha))
        )

    @property
    def frame(self): return self.model.get_frame()
    @property
    def status_text(self): return self.model.get_status_text()
    
    def get_player_field(self, pid): return self.model.get_player_state(pid).field
    def get_player_score(self, pid): return self.model.get_player_state(pid).score
    def get_player_ojama(self, pid): 
        s = self.model.get_player_state(pid)
        return s.non_active_ojama, s.active_ojama
    def get_chain_count(self, pid):
        """Returns (current_chain, last_chain). current_chain > 0 during chain pop.
        last_chain is the result of the last completed chain (0 if no chain yet).
        """
        s = self.model.get_player_state(pid)
        return s.chain_count, s.last_chain_count
    
    def get_next_piece(self, pid, offset): return self.model.get_piece(pid, offset)
