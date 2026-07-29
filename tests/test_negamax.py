import sys
import time
from pathlib import Path

# Add native/dist to sys.path
_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def main():
    print("=== Testing Negamax Adversarial Search ===")
    
    # 1. Initialize Match
    seed = 42
    match = p.PuyotanMatch(seed)
    match.start()
    
    # 2. Config Negamax
    cfg = p.NegamaxConfig()
    cfg.depth = 4          # 4 decision turns ahead (2 for P1, 2 for P2)
    cfg.candidate_n = 5    # Top 5 candidate moves
    
    # Load base VS beam config
    cfg_path = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
    cfg.vs_config = p.load_vs_config(cfg_path)
    cfg.vs_config.beam_width = 200
    cfg.vs_config.look_ahead = 3
    
    print(f"Match frame: {match.frame}, Status: {match.status}")
    print(f"Negamax Config: depth={cfg.depth}, candidate_n={cfg.candidate_n}")
    
    # 3. Perform Negamax Search for P1 (index 0)
    t0 = time.time()
    res = p.negamax_search(match, 0, cfg)
    elapsed = time.time() - t0
    
    print(f"\n--- Search Result ---")
    print(f"Time taken: {elapsed * 1000:.2f} ms")
    print(f"Best Action Index: {res.best_action}")
    print(f"Best Action Details: {p.get_rl_action(res.best_action)}")
    print(f"Best Evaluation: {res.best_eval:.2f}")
    print("\nCandidate Evaluations at Root:")
    for act_idx, score in res.candidate_evals:
        action = p.get_rl_action(act_idx)
        print(f"  Action idx={act_idx:2d} (col={action.x}, rot={action.rotation}): eval={score:10.2f}")
        
    print("\nSUCCESS: Negamax search executed clean!")

if __name__ == "__main__":
    main()
