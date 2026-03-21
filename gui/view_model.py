import puyotan_native as p
from PyQt6.QtCore import QObject, pyqtSignal, QElapsedTimer
from . import config

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
    Encapsulates Game logic, input-to-engine mapping, and timing.
    Emits signals when the view should be updated.
    """
    state_changed = pyqtSignal()
    game_over = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.players = [PlayerPresentationState(), PlayerPresentationState()]
        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_step_time = self.timer.elapsed()
        
        # ColorMap localized here without Pygame
        self.p_colors = {
            p.Cell.Red: config.COLORS["Red"],
            p.Cell.Green: config.COLORS["Green"],
            p.Cell.Blue: config.COLORS["Blue"],
            p.Cell.Yellow: config.COLORS["Yellow"],
            p.Cell.Ojama: config.COLORS["Ojama"],
            p.Cell.Empty: config.COLORS["Empty"]
        }
        self.update_presentation()

    def update(self):
        """Main update loop for logic and timing (called by QTimer)."""
        if not self.model.is_playing():
            return

        current_time = self.timer.elapsed()
        state_was_changed = False

        # 1. Check if we can step (all ready or rigid)
        if self.model.can_step():
            if current_time - self.last_step_time >= config.VIRTUAL_FRAME_INTERVAL_MS:
                if self.model.step():
                    self.last_step_time = current_time
                    state_was_changed = True
                    # Reset positions for players who just got a new decision point
                    for pid in [0, 1]:
                        if not self.model.has_pending_action(pid):
                            self.reset_player_input(pid)
        else:
            # At least one player needs to confirm
            for pid in [0, 1]:
                if not self.model.has_pending_action(pid) and self.players[pid].confirmed:
                    # Actually push action to model
                    action = p.Action(p.ActionType.PUT, self.players[pid].x, self.players[pid].rotation)
                    if self.model.set_action(pid, action):
                        state_was_changed = True

        if state_was_changed:
            self.update_presentation()
            if not self.model.is_playing():
                self.game_over.emit(self.model.get_status_text())

    def update_presentation(self):
        """Refresh presentation-only state from the model."""
        for pid in [0, 1]:
            player_state = self.model.get_player_state(pid)
            pres = self.players[pid]
            
            pres.rigid_frames = sum(ah.remaining_frame for ah in player_state.action_histories)
            pres.has_decision = not self.model.has_pending_action(pid)
            
            # If they just became rigid, force confirmed off
            if not pres.has_decision:
                pres.confirmed = False

            # Calculate ghost colors
            piece = self.model.get_piece(pid, 0)
            pres.ghost_color_axis = self._blend_ghost(self.p_colors.get(piece.axis, (255,255,255)))
            pres.ghost_color_sub = self._blend_ghost(self.p_colors.get(piece.sub, (255,255,255)))
            
        self.state_changed.emit()

    def move_player(self, pid, dx):
        if not self.model.is_playing():
            return
        if not self.players[pid].confirmed and self.players[pid].has_decision:
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
        if not self.players[pid].confirmed and self.players[pid].has_decision:
            rot_order = [p.Rotation.Up, p.Rotation.Right, p.Rotation.Down, p.Rotation.Left]
            idx = rot_order.index(self.players[pid].rotation)
            new_rot = rot_order[(idx + direction) % 4]
            self.players[pid].rotation = new_rot
            
            # Apply wall kicks
            if new_rot == p.Rotation.Left and self.players[pid].x == 0:
                self.players[pid].x = 1
            elif new_rot == p.Rotation.Right and self.players[pid].x == 5:
                self.players[pid].x = 4
                
            self.state_changed.emit()

    def confirm_player(self, pid):
        if not self.model.is_playing():
            return
        if self.players[pid].has_decision and not self.players[pid].confirmed:
            self.players[pid].confirmed = True
            # Try to push to model immediately
            action = p.Action(p.ActionType.PUT, self.players[pid].x, self.players[pid].rotation)
            self.model.set_action(pid, action)
            self.update_presentation()

    def reset_player_input(self, pid):
        self.players[pid].x = 2
        self.players[pid].rotation = p.Rotation.Up
        self.players[pid].confirmed = False

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

    # Simple getters for the view
    @property
    def frame(self): return self.model.get_frame()
    @property
    def status_text(self): return self.model.get_status_text()
    
    def get_player_field(self, pid): return self.model.get_player_state(pid).field
    def get_player_score(self, pid): return self.model.get_player_state(pid).score
    def get_player_ojama(self, pid): 
        s = self.model.get_player_state(pid)
        return s.non_active_ojama, s.active_ojama
    
    def get_next_piece(self, pid, offset): return self.model.get_piece(pid, offset)
