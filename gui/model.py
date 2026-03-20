import puyotan_native as p

class GameModel:
    """
    Wraps the puyotan_native.PuyotanMatch engine.
    Provides methods to retrieve necessary state for rendering
    and passing actions to the engine.
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.restart()

    def get_piece(self, player_id, index_offset):
        return self.match.getPiece(player_id, index_offset)
        
    def restart(self):
        self.match = p.PuyotanMatch(self.seed)
        self.match.start()

    def get_player_state(self, player_id: int):
        return self.match.getPlayer(player_id)

    def get_frame(self) -> int:
        return self.match.frame

    def get_status(self):
        return self.match.status

    def get_status_text(self) -> str:
        return self.match.status_text

    def set_action(self, player_id: int, action: p.Action) -> bool:
        """
        Attempts to set the action for the current frame.
        Returns True if successful, False if the player cannot act yet.
        """
        return self.match.setAction(player_id, action)

    def can_step(self):
        return self.match.canStepNextFrame()

    def step(self):
        """
        Advances the match by one frame if both inputs are ready.
        Returns True if a frame was advanced.
        """
        if self.match.canStepNextFrame():
            self.match.stepNextFrame()
            return True
        return False
        
    def is_playing(self) -> bool:
        return self.match.status == p.MatchStatus.PLAYING

    def has_pending_action(self, player_id: int) -> bool:
        """
        Check if the player has an action submitted for the *current* frame.
        """
        player = self.get_player_state(player_id)
        return player.action_histories[self.match.frame & 255].action.type != p.ActionType.NONE
