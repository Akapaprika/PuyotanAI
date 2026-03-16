import pygame
import puyotan_native as p
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

class PuyotanViewModel:
    """
    The ViewModel for the Puyotan GUI.
    Encapsulates Game logic, input-to-engine mapping, and timing.
    """
    def __init__(self, model):
        self.model = model
        self.players = [PlayerPresentationState(), PlayerPresentationState()]
        self.last_step_time = pygame.time.get_ticks()
        self.update_presentation()

    def update(self, current_time):
        """Main update loop for logic and timing."""
        if not self.model.is_playing():
            return

        # 1. Check if we can step (all ready or rigid)
        if self.model.can_step():
            if current_time - self.last_step_time >= config.VIRTUAL_FRAME_INTERVAL:
                if self.model.step():
                    self.last_step_time = current_time
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
                        # We don't reset confirmed here; update_presentation will handle it
                        pass

        self.update_presentation()

    def update_presentation(self):
        """Refresh presentation-only state from the model."""
        for pid in [0, 1]:
            player_state = self.model.get_player_state(pid)
            pres = self.players[pid]
            
            pres.rigid_frames = sum(ah.remaining_frame for ah in player_state.action_histories.values())
            pres.has_decision = not self.model.has_pending_action(pid)
            
            # If they just became rigid, force confirmed off
            if not pres.has_decision:
                pres.confirmed = False

            # Calculate ghost colors
            piece = self.model.get_piece(pid, 0)
            pres.ghost_color_axis = self._blend_ghost(config.COLOR_MAP.get(piece.axis))
            pres.ghost_color_sub = self._blend_ghost(config.COLOR_MAP.get(piece.sub))

    def move_player(self, pid, dx):
        if not self.players[pid].confirmed and self.players[pid].has_decision:
            self.players[pid].x = max(0, min(5, self.players[pid].x + dx))

    def rotate_player(self, pid, dir):
        if not self.players[pid].confirmed and self.players[pid].has_decision:
            rot_order = [p.Rotation.Up, p.Rotation.Right, p.Rotation.Down, p.Rotation.Left]
            idx = rot_order.index(self.players[pid].rotation)
            self.players[pid].rotation = rot_order[(idx + dir) % 4]

    def confirm_player(self, pid):
        if self.players[pid].has_decision:
            self.players[pid].confirmed = True

    def reset_player_input(self, pid):
        self.players[pid].x = 2
        self.players[pid].rotation = p.Rotation.Up
        self.players[pid].confirmed = False

    def restart(self):
        self.model.restart()
        for pid in [0, 1]:
            self.reset_player_input(pid)
        self.last_step_time = pygame.time.get_ticks()
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
