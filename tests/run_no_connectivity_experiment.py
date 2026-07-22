import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../native/dist")))
import puyotan_native as puyotan

def run_no_connectivity_experiment(beam_width=3000, look_ahead=10, num_games=20):
    print(f"=== Deep Search Experiment (No Connectivity Bonus: 0.0) ===")
    print(f"Config: beam_width={beam_width}, look_ahead={look_ahead}, num_games={num_games}")
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../gui/beam_config.json"))
    
    p1_cfg = puyotan.load_vs_config(config_path)
    p1_cfg.enable_attack_search = True
    p1_cfg.beam_width = beam_width
    p1_cfg.look_ahead = look_ahead
    p1_cfg.eval_weights.connectivity_bonus = 0.0

    p2_cfg = puyotan.load_vs_config(config_path)
    p2_cfg.enable_attack_search = False
    p2_cfg.beam_width = beam_width
    p2_cfg.look_ahead = look_ahead
    p2_cfg.eval_weights.connectivity_bonus = 0.0

    seeds = [i + 1 for i in range(num_games)]

    start_time = time.time()
    results = puyotan.simulate_vs_matches_parallel(p1_cfg, p2_cfg, seeds)
    elapsed = time.time() - start_time

    p1_wins = sum(1 for r in results if r.status == puyotan.MatchStatus.WIN_P1)
    p2_wins = sum(1 for r in results if r.status == puyotan.MatchStatus.WIN_P2)
    avg_frames = sum(r.total_frames for r in results) / num_games
    max_chain_p1 = max(r.max_chain_p1 for r in results)
    max_chain_p2 = max(r.max_chain_p2 for r in results)

    print(f"\n--- Results ({elapsed:.2f}s total) ---")
    print(f"Player 1 (New AI: Gaze & Sub-chain ON, No Conn) : {p1_wins} wins ({(p1_wins/num_games)*100:.1f}%)")
    print(f"Player 2 (Old AI: Gaze & Sub-chain OFF, No Conn): {p2_wins} wins ({(p2_wins/num_games)*100:.1f}%)")
    print(f"Average Frames per Game: {avg_frames:.1f}")
    print(f"Max Chain: P1={max_chain_p1}, P2={max_chain_p2}")

if __name__ == "__main__":
    run_no_connectivity_experiment(3000, 10, 20)
