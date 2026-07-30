import sys
import time
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def run_match(seed: int, mode: str = "abs", max_frames: int = 10000):
    """
    mode: 'abs' or 'negamax' or 'vsbeam'
    """
    match = p.PuyotanMatch(seed)
    match.start()
    
    cfg_path = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
    vs_cfg = p.load_vs_config(cfg_path)
    vs_cfg.enable_attack_search = True

    p1_session = p.BeamSearchSession()
    p2_session = p.BeamSearchSession()

    abs_cfg = p.load_abs_config(cfg_path)
    nega_cfg = p.load_negamax_config(cfg_path)

    while match.status == p.MatchStatus.PLAYING and match.frame < max_frames:
        mask = match.getDecisionMask()
        if mask != 0:
            if mask & 1: # Player 0 turn
                if mode == "abs":
                    res = p.abs_search(match, 0, abs_cfg)
                    match.setAction(0, p.get_rl_action(res.best_action))
                elif mode == "negamax":
                    res = p.negamax_search(match, 0, nega_cfg)
                    match.setAction(0, p.get_rl_action(res.best_action))
                else:
                    action_idx, _ = p.vs_beam_search(match.getPlayer(0), match.getTsumo(), vs_cfg, p1_session)
                    match.setAction(0, p.get_rl_action(action_idx))

            if mask & 2: # Player 1 turn (VsBeam AI)
                action_idx, _ = p.vs_beam_search(match.getPlayer(1), match.getTsumo(), vs_cfg, p2_session)
                match.setAction(1, p.get_rl_action(action_idx))

        match.stepNextFrame()

    turns_p1 = match.getPlayer(0).active_next_pos
    turns_p2 = match.getPlayer(1).active_next_pos
    return match.status, match.getPlayer(0).score, match.getPlayer(1).score, turns_p1, turns_p2

def main():
    mode = "abs"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    agent_name = "ABS AI (depth=10)" if mode == "abs" else "Negamax AI"
    print("=========================================================")
    print(f"      VS AI Benchmark (20 Games): {agent_name} vs VsBeam   ")
    print("=========================================================")

    num_games = 20
    seeds = [100 + i for i in range(num_games)]

    print(f"Running {num_games} matches with {agent_name} vs VsBeam...")
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_time = 0.0

    for i, seed in enumerate(seeds):
        t0 = time.time()
        st, s1, s2, t1, t2 = run_match(seed, mode=mode)
        elapsed = time.time() - t0
        total_time += elapsed

        winner = "DRAW"
        if st == p.MatchStatus.WIN_P1:
            p1_wins += 1
            winner = f"P1 ({agent_name})"
        elif st == p.MatchStatus.WIN_P2:
            p2_wins += 1
            winner = "P2 (VsBeam)"
        else:
            draws += 1

        print(f"Game {i+1:2d} | Seed: {seed} | Winner: {winner:<18} | Score P1: {s1:6d} vs P2: {s2:6d} | Turns: {t1:2d} (P1) / {t2:2d} (P2) | Time: {elapsed:.2f}s")

    print("\n---------------------------------------------------------")
    print(f"Total Matches: {num_games}")
    print(f"P1 ({agent_name}) Wins : {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"P2 (VsBeam) Wins      : {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws                 : {draws}")
    print(f"Avg Match Time        : {total_time/num_games:.2f}s")
    print("=========================================================")

if __name__ == "__main__":
    main()
