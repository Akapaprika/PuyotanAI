"""
tests/test_solo_regression.py

ソロAIのリグレッションテスト（回帰テスト）スクリプト。
固定標準パラメータ: Seed=1, BeamWidth=15000, LookAhead=25, DBS=6 で50手分実行し、
ゴールドマスター（基準データ）と100%完全一致するかどうかを厳格に判定します。
"""
import os
import sys
import json
from pathlib import Path

# C++ モジュール (puyotan_native) のパス設定
_PROJECT_ROOT = Path(__file__).parent.parent
_NATIVE_DIST = _PROJECT_ROOT / "native" / "dist"
_GOLDEN_PATH = _PROJECT_ROOT / "tests" / "golden_solo_seed1_50steps.json"

if str(_NATIVE_DIST) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIST))

import puyotan_native as p

# 回帰テスト用 固定標準パラメータ
BENCH_BEAM_WIDTH = 15000
BENCH_LOOK_AHEAD = 25
BENCH_DBS = 6


def run_50step_simulation(seed: int = 1, num_moves: int = 50):
    """Seed=1 で固定パラメータ (BW=15000, LA=25) で50手分のソロビームサーチを実行し、記録を返す。"""
    match = p.PuyotanMatch(seed)
    match.start()
    pass_act = p.Action(p.ActionType.PASS, 0, p.Rotation.Up)

    # 固定設定の構築
    cfg = p.SoloBeamConfig()
    cfg.beam_width = BENCH_BEAM_WIDTH
    cfg.look_ahead = BENCH_LOOK_AHEAD
    cfg.dbs_max_similar = BENCH_DBS
    cfg.full_beam_depth = 2
    cfg.min_beam_width_ratio = 1.0  # 基準はテーパリング減衰なし（均一幅）

    records = []

    print(f"[Regression Test] Seed={seed}, BW={BENCH_BEAM_WIDTH}, LA={BENCH_LOOK_AHEAD}: Running {num_moves} moves...")

    for move in range(1, num_moves + 1):
        player = match.getPlayer(0)
        tsumo = match.getTsumo()

        # Pure search API (solo_beam_search)
        act_idx, score = p.solo_beam_search(player, tsumo, cfg)

        action = p.get_rl_action(act_idx)

        record = {
            "move": move,
            "act_idx": int(act_idx),
            "x": int(action.x),
            "rotation": int(action.rotation),
            "score": round(float(score), 2)
        }
        records.append(record)

        print(f"  Move {move:2d}/50: Action={act_idx:2d} (x={action.x}, rot={int(action.rotation)}) | EvalScore={score:.1f}")

        # Step match forward
        match.setAction(0, action)
        while match.status == p.MatchStatus.PLAYING:
            mask = match.getDecisionMask()
            if mask & 1:
                break
            if mask & 2:
                match.setAction(1, pass_act)
            match.stepNextFrame()

        if match.status != p.MatchStatus.PLAYING:
            print(f"  [Notice] Match finished at move {move} (Status != PLAYING)")
            break

    return records


def generate_golden():
    """Generates the golden master reference dataset."""
    records = run_50step_simulation(seed=1, num_moves=50)
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Golden master saved: {_GOLDEN_PATH}")
    return records


def verify_against_golden():
    """Verifies 100% exact match against golden master."""
    if not _GOLDEN_PATH.exists():
        print(f"\n[Notice] Golden master not found. Generating...")
        return generate_golden()

    with open(_GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    current = run_50step_simulation(seed=1, num_moves=50)

    print("\n" + "=" * 60)
    print("  Solo AI Regression Test (Golden Master Comparison)")
    print("=" * 60)

    mismatches = 0
    total_moves = min(len(golden), len(current))

    for i in range(total_moves):
        g = golden[i]
        c = current[i]

        is_match = (g["act_idx"] == c["act_idx"] and g["x"] == c["x"] and g["rotation"] == c["rotation"])
        if is_match:
            status = "OK (MATCH)"
        else:
            status = f"NG (MISMATCH! Golden: act={g['act_idx']} x={g['x']} rot={g['rotation']} | Current: act={c['act_idx']} x={c['x']} rot={c['rotation']})"
            mismatches += 1

        print(f"Move {i+1:2d}: {status}")

    print("=" * 60)

    if mismatches == 0 and len(golden) == len(current):
        print(" [PASS] All 50 moves 100% MATCHED! No solo AI regression.")
        return True
    else:
        print(f" [FAIL] Found {mismatches} mismatches!")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_golden()
    else:
        verify_against_golden()
