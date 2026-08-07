import os
import sys
import time

# Ensure dist directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../native/dist")))

import puyotan_native as puyotan

def run_vs_comparison(num_games=100):
    print(f"=== Running VS Match Simulation: New VS AI (P1) vs Old VS AI (P2) ===")
    print(f"Number of games: {num_games}")

    # Load baseline VS config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../gui/beam_config.json"))
    
    # Configure Player 1 (New AI with attack search & gaze enabled)
    p1_cfg = puyotan.load_vs_config(config_path)
    p1_cfg.enable_attack_search = True

    # Configure Player 2 (Old AI with attack search disabled)
    p2_cfg = puyotan.load_vs_config(config_path)
    p2_cfg.enable_attack_search = False

    seeds = [i + 1 for i in range(num_games)]

    start_time = time.time()
    results = puyotan.simulate_vs_match_parallel(p1_cfg, p2_cfg, seeds) if hasattr(puyotan, "simulate_vs_match_parallel") else puyotan.simulate_vs_matches_parallel(p1_cfg, p2_cfg, seeds)
    elapsed = time.time() - start_time

    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_frames = 0
    max_chain_p1_all = 0
    max_chain_p2_all = 0

    for r in results:
        total_frames += r.total_frames
        max_chain_p1_all = max(max_chain_p1_all, r.max_chain_p1)
        max_chain_p2_all = max(max_chain_p2_all, r.max_chain_p2)
        if r.status == puyotan.MatchStatus.WIN_P1:
            p1_wins += 1
        elif r.status == puyotan.MatchStatus.WIN_P2:
            p2_wins += 1
        else:
            draws += 1

    win_rate_p1 = (p1_wins / num_games) * 100.0
    win_rate_p2 = (p2_wins / num_games) * 100.0
    avg_frames = total_frames / num_games

    print(f"\n--- Simulation Results ({elapsed:.2f} seconds total) ---")
    print(f"Player 1 (New AI: Attack Search & Gaze ON) : {p1_wins} wins ({win_rate_p1:.1f}%)")
    print(f"Player 2 (Old AI: Attack Search & Gaze OFF): {p2_wins} wins ({win_rate_p2:.1f}%)")
    print(f"Draws: {draws}")
    print(f"Average Frames per Game: {avg_frames:.1f}")
    print(f"Max Chain Achieved: P1={max_chain_p1_all}, P2={max_chain_p2_all}")

if __name__ == "__main__":
    run_vs_comparison(100)
