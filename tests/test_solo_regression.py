"""
tests/test_solo_regression.py

ソロAIのリグレッションテスト（回帰テスト）スクリプト。
Seed=1、LookAhead=25、BeamWidth=15000（ソロ全開モード）で50手分実行し、
ゴールドマスター（マスターデータ）と100%完全一致するかどうかを厳格に判定します。
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# C++ モジュール (puyotan_native) のパス設定
_PROJECT_ROOT = Path(__file__).parent.parent
_NATIVE_DIST = _PROJECT_ROOT / "native" / "dist"
_CONFIG_PATH = str(_PROJECT_ROOT / "native" / "resources" / "beam_config.json")
_GOLDEN_PATH = _PROJECT_ROOT / "tests" / "golden_solo_seed1_50steps.json"

if str(_NATIVE_DIST) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIST))

import puyotan_native as p

def run_50step_simulation(seed: int = 1, num_moves: int = 50):
    """Seed=1 で50手分のソロビームサーチを実行し、(move, act_idx, x, rotation, score) の記録を返す。"""
    match = p.PuyotanMatch(seed)
    match.start()
    pass_act = p.Action(p.ActionType.PASS, 0, p.Rotation.Up)

    records = []

    print(f"[Regression Test] Seed={seed} で {num_moves}手分のソロビームサーチを実行中...")

    for move in range(1, num_moves + 1):
        player = match.getPlayer(0)
        tsumo  = match.getTsumo()

        # ソロ全開ビームサーチを実行 (beam_width=-1, look_ahead=-1 -> beam_config.json のデフォルト使用)
        act_idx, score = p.beam_search_action(
            player, tsumo, _CONFIG_PATH,
            beam_width=-1, look_ahead=-1,
            is_solo=True, is_stagnated=False
        )

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

        # 手を反映してゲームを進める
        match.setAction(0, action)
        while match.status == p.MatchStatus.PLAYING:
            mask = match.getDecisionMask()
            if mask & 1:
                break
            if mask & 2:
                match.setAction(1, pass_act)
            match.stepNextFrame()

        if match.status != p.MatchStatus.PLAYING:
            print(f"  [Notice] Move {move} でゲーム終了（MatchStatus != PLAYING）")
            break

    return records

def generate_golden():
    """現在のコード動作を正しいゴールドマスターとして保存する。"""
    records = run_50step_simulation(seed=1, num_moves=50)
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] ゴールドマスター（基準データ）を生成・保存しました: {_GOLDEN_PATH}")
    return records

def verify_against_golden():
    """既存のゴールドマスターと現在の動作を1手ずつ比較し、100%完全一致するか検証する。"""
    if not _GOLDEN_PATH.exists():
        print(f"\n[Notice] ゴールドマスターファイルが見つかりません。まず基準データを生成します。")
        return generate_golden()

    with open(_GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    current = run_50step_simulation(seed=1, num_moves=50)

    print("\n" + "=" * 60)
    print("  ソロAI リグレッションテスト結果（ゴールドマスター比較）")
    print("=" * 60)

    mismatches = 0
    total_moves = min(len(golden), len(current))

    for i in range(total_moves):
        g = golden[i]
        c = current[i]

        is_match = (g["act_idx"] == c["act_idx"] and g["x"] == c["x"] and g["rotation"] == c["rotation"])
        if is_match:
            status = "OK (一致)"
        else:
            status = f"NG (不一致! Golden: act={g['act_idx']} x={g['x']} rot={g['rotation']} | Current: act={c['act_idx']} x={c['x']} rot={c['rotation']})"
            mismatches += 1

        print(f"Move {i+1:2d}: {status}")

    print("=" * 60)

    if mismatches == 0 and len(golden) == len(current):
        print(" [PASS] 50手すべて 100% 完全一致しました！ソロAIの機能回帰はありません。")
        return True
    else:
        print(f" [FAIL] 不一致が {mismatches} 件検出されました！ソロAIの動作が変化しています。")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_golden()
    else:
        verify_against_golden()
