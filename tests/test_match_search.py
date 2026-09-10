import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_NATIVE_DIST = _PROJECT_ROOT / "native" / "dist"
if str(_NATIVE_DIST) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIST))

import puyotan_native as p
from ai.beam_agents import MatchBeamAgent, SoloBeamAgent
from ai.config import CONFIG_PATH

def test_match_config():
    print("=== Test 1: MatchBeamConfig ===")
    cfg = p.MatchBeamConfig()
    assert cfg.beam_width == 5000, f"Expected 5000, got {cfg.beam_width}"
    assert cfg.look_ahead == 5, f"Expected 5, got {cfg.look_ahead}"
    print(f"Default config: beam_width={cfg.beam_width}, look_ahead={cfg.look_ahead}")

    loaded_cfg = p.load_match_config(CONFIG_PATH)
    assert loaded_cfg.beam_width > 0
    assert loaded_cfg.look_ahead > 0
    print(f"Loaded config from JSON: beam_width={loaded_cfg.beam_width}, look_ahead={loaded_cfg.look_ahead}")
    print("[PASS] MatchBeamConfig test passed.\n")


def test_match_beam_search_execution():
    print("=== Test 2: match_beam_search Execution & Speed ===")
    match = p.PuyotanMatch(1)
    match.start()
    match.stepUntilDecision()

    cfg = p.load_match_config(CONFIG_PATH)

    print("Running match_beam_search(match, my_id=0, cfg)...")
    start_t = time.perf_counter()
    act_idx, score = p.match_beam_search(match, 0, cfg)
    elapsed = time.perf_counter() - start_t

    action = p.get_rl_action(act_idx)
    print(f"Result: act_idx={act_idx} (x={action.x}, rot={action.rotation}), score={score}")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    assert 0 <= act_idx < 22, f"Invalid act_idx: {act_idx}"
    print("[PASS] match_beam_search execution test passed.\n")


from gui.model import GameModel


def test_match_agent_vs_solo():
    print("=== Test 3: MatchBeamAgent Match Simulation (5 moves) ===")
    game = GameModel(1)

    p0 = MatchBeamAgent()
    p1 = SoloBeamAgent()

    # 5手シミュレーション
    for step in range(5):
        mask = game.match.getDecisionMask()
        if game.match.status != p.MatchStatus.PLAYING:
            break

        print(f"Step {step+1}: DecisionMask={mask}")
        
        # P0
        if mask & 1:
            act0 = None
            while act0 is None:
                act0 = p0.get_action(game, 0)
                time.sleep(0.02)
            print(f"  P0 (MatchBeam) chose action: x={act0.x}, rot={int(act0.rotation)}")
            game.set_action(0, act0)

        # P1
        if mask & 2:
            act1 = None
            while act1 is None:
                act1 = p1.get_action(game, 1)
                time.sleep(0.02)
            print(f"  P1 (SoloBeam) chose action: x={act1.x}, rot={int(act1.rotation)}")
            game.set_action(1, act1)

        game.match.stepUntilDecision()

    print("[PASS] MatchBeamAgent simulation test passed.\n")


def run_all():
    test_match_config()
    test_match_beam_search_execution()
    test_match_agent_vs_solo()


if __name__ == "__main__":
    run_all()
    print("All match search tests PASSED successfully!")
