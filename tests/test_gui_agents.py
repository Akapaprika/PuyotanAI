"""
tests/test_gui_agents.py
gui/agents.py の各 Agent クラス (BeamSearchAgent, VsBeamSearchAgent, NegamaxAgent) の動作検証。
"""
import sys
import time
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "native" / "resources" / "beam_config.json")

if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import puyotan_native as p
from gui.agents import BeamSearchAgent, VsBeamSearchAgent, NegamaxAgent

class DummyGameModel:
    """GUI の ViewModel / GameModel のインターフェースをシミュレートするダミークラス"""
    def __init__(self, seed: int = 1):
        self.match = p.PuyotanMatch(seed)
        self.match.start()

    def get_player_state(self, player_id: int):
        return self.match.getPlayer(player_id)

def test_gui_agent_instantiation_and_action():
    model = DummyGameModel(seed=1)

    agent_solo = BeamSearchAgent(beam_width=100, look_ahead=2)
    agent_vs = VsBeamSearchAgent(beam_width=100, look_ahead=2)
    agent_nega = NegamaxAgent(depth=2, candidate_n=2, beam_width=100, look_ahead=2)

    # Agent は非同期スレッドで探索を行うため、完了まで短時間ループ待ち
    for agent in [agent_solo, agent_vs, agent_nega]:
        act = None
        for _ in range(200):
            act = agent.get_action(model, 0, pres=None)
            if act is not None:
                break
            time.sleep(0.01)
        assert act is not None, f"Agent {agent} failed to return an action within timeout"

def run_all():
    print("Running test_gui_agents...")
    test_gui_agent_instantiation_and_action()
    print("  [PASS] test_gui_agents")

if __name__ == "__main__":
    run_all()
