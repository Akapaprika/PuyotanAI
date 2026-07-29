import sys
import time
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def run_match(seed: int, p1_negamax: bool, depth: int = 4, max_frames: int = 10000):
    match = p.PuyotanMatch(seed)
    match.start()
    
    cfg_path = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
    vs_cfg = p.load_vs_config(cfg_path)
    vs_cfg.beam_width = 3000
    vs_cfg.look_ahead = 3

    p1_session = p.BeamSearchSession()
    p2_session = p.BeamSearchSession()

    nega_cfg = p.NegamaxConfig()
    nega_cfg.depth = depth
    nega_cfg.candidate_n = 22
    nega_cfg.vs_config = vs_cfg

    p1_nodes = 0
    p2_nodes = 0

    while match.status == p.MatchStatus.PLAYING and match.frame < max_frames:
        mask = match.getDecisionMask()
        if mask != 0:
            if mask & 1: # Player 0 turn
                if p1_negamax:
                    res = p.negamax_search(match, 0, nega_cfg)
                    match.setAction(0, p.get_rl_action(res.best_action))
                else:
                    action_idx, _ = p.vs_beam_search(match.getPlayer(0), match.getTsumo(), vs_cfg, p1_session)
                    match.setAction(0, p.get_rl_action(action_idx))

            if mask & 2: # Player 1 turn
                # Player 1 is always conventional VS Beam Search
                action_idx, _ = p.vs_beam_search(match.getPlayer(1), match.getTsumo(), vs_cfg, p2_session)
                match.setAction(1, p.get_rl_action(action_idx))

        match.stepNextFrame()

    return match.status, match.getPlayer(0).score, match.getPlayer(1).score, match.frame

def main():
    print("=========================================================")
    print("      VS AI Benchmark: Negamax AI  vs  VsBeam AI        ")
    print("=========================================================")

    num_games = 10
    seeds = [100 + i for i in range(num_games)]

    print(f"Running {num_games} matches with Negamax depth=4...")
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_time = 0.0

    for i, seed in enumerate(seeds):
        t0 = time.time()
        st, s1, s2, frames = run_match(seed, p1_negamax=True, depth=4)
        elapsed = time.time() - t0
        total_time += elapsed

        winner = "DRAW"
        if st == p.MatchStatus.WIN_P1:
            p1_wins += 1
            winner = "P1 (Negamax)"
        elif st == p.MatchStatus.WIN_P2:
            p2_wins += 1
            winner = "P2 (VsBeam)"
        else:
            draws += 1

        print(f"Game {i+1:2d} | Seed: {seed} | Winner: {winner:<13} | Score P1: {s1:6d} vs P2: {s2:6d} | Frames: {frames:5d} | Time: {elapsed:.2f}s")

    print("\n---------------------------------------------------------")
    print(f"Total Matches: {num_games}")
    print(f"P1 (Negamax) Wins : {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"P2 (VsBeam) Wins  : {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws             : {draws}")
    print(f"Avg Match Time    : {total_time/num_games:.2f}s")
    print("=========================================================")

if __name__ == "__main__":
    main()
