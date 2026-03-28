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
        self.match.stepUntilDecision()  # Advance to first PUT decision point

    def get_player_state(self, player_id: int):
        return self.match.getPlayer(player_id)

    def get_frame(self) -> int:
        return self.match.frame

    def get_status(self):
        return self.match.status

    def get_status_text(self) -> str:
        status_map = {
            p.MatchStatus.READY:   "Ready",
            p.MatchStatus.PLAYING: "Playing",
            p.MatchStatus.WIN_P1:  "Player 1 Wins!",
            p.MatchStatus.WIN_P2:  "Player 2 Wins!",
            p.MatchStatus.DRAW:    "Draw!",
        }
        return status_map.get(self.match.status, "Unknown")

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

    def get_decision_mask(self) -> int:
        """
        Returns a bitmask of players that need to submit a PUT action.
        0 = no input needed (auto-frames running)
        1 = P1 needs PUT
        2 = P2 needs PUT
        3 = both need PUT
        """
        return self.match.getDecisionMask()
