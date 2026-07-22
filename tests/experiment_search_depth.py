import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../native/dist")))
import puyotan_native as puyotan

def run_experiment(name, beam_width, look_ahead, num_games=100):
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../gui/beam_config.json"))
    
    p1_cfg = puyotan.load_vs_config(config_path)
    p1_cfg.enable_attack_search = True
    p1_cfg.beam_width = beam_width
    p1_cfg.look_ahead = look_ahead

    p2_cfg = puyotan.load_vs_config(config_path)
    p2_cfg.enable_attack_search = False
    p2_cfg.beam_width = beam_width
    p2_cfg.look_ahead = look_ahead

    seeds = [i + 1 for i in range(num_games)]

    start_time = time.time()
    results = puyotan.simulate_vs_matches_parallel(p1_cfg, p2_cfg, seeds)
    elapsed = time.time() - start_time

    p1_wins = sum(1 for r in results if r.status == puyotan.MatchStatus.WIN_P1)
    p2_wins = sum(1 for r in results if r.status == puyotan.MatchStatus.WIN_P2)
    avg_frames = sum(r.total_frames for r in results) / num_games

    print(f"[{name}] BW={beam_width}, LA={look_ahead} -> P1 Wins: {p1_wins}%, P2 Wins: {p2_wins}%, Avg Frames: {avg_frames:.1f}, Time: {elapsed:.2f}s")

if __name__ == "__main__":
    print("=== Running Search Depth / Width Experiments (100 Games Each) ===")
    run_experiment("Baseline  ", 500, 3)
    run_experiment("Deeper LA ", 500, 4)
    run_experiment("Wider Beam", 1000, 3)
    run_experiment("Deep+Wide ", 1000, 4)
