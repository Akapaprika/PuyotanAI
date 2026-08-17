"""
tests/test_engine_match.py
C++ PuyotanMatch (ゲーム進行・状態遷移・おじゃまぷよ) の単体テスト。
"""
import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def test_match_lifecycle():
    match = p.PuyotanMatch(12345)
    assert match.status == p.MatchStatus.READY
    match.start()
    assert match.status == p.MatchStatus.PLAYING
    assert match.frame >= 1

    p1 = match.getPlayer(0)
    p2 = match.getPlayer(1)
    assert p1.score == 0
    assert p2.score == 0

def test_match_frame_stepping_and_action():
    """着手設定とフレームステップ進行、意思決定マスクの検証"""
    match = p.PuyotanMatch(42)
    match.start()
    
    # Player 0 に着手を設定
    act = p.Action(p.ActionType.PUT, 2, p.Rotation.Up)
    pass_act = p.Action(p.ActionType.PASS, 0, p.Rotation.Up)
    
    match.setAction(0, act)
    match.setAction(1, pass_act)
    
    initial_frame = match.frame
    # 10フレーム進める
    for _ in range(10):
        if match.canStepNextFrame():
            match.stepNextFrame()
            
    assert match.frame > initial_frame, "stepNextFrame によりフレームが進んでいません"

def run_all():
    print("Running test_engine_match...")
    test_match_lifecycle()
    test_match_frame_stepping_and_action()
    print("  [PASS] test_engine_match")

if __name__ == "__main__":
    run_all()
