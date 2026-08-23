"""
tests/test_board_and_chain.py
C++ Core (BitBoard, Board, Chain, Scorer, Gravity) の単体テスト。
"""
import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def test_bitboard():
    bb = p.BitBoard()
    assert bb.empty()
    bb.set(0, 0)
    assert not bb.empty()
    assert bb.get(0, 0)
    assert bb.popcount() == 1

def test_board_and_gravity():
    board = p.Board()
    board.set(0, 0, p.Cell.Red)
    assert board.get(0, 0) == p.Cell.Red

def test_get_active_puyos():
    board = p.Board()
    # 5色それぞれセット
    board.set(0, 0, p.Cell.Red)
    board.set(1, 2, p.Cell.Green)
    board.set(2, 4, p.Cell.Blue)
    board.set(3, 6, p.Cell.Yellow)
    board.set(5, 12, p.Cell.Ojama)

    active = board.get_active_puyos()
    assert len(active) == 5

    # 辞書化して全色・全座標が一致するか確認
    active_map = {(puyo.x, puyo.y): puyo.color for puyo in active}
    assert active_map[(0, 0)] == p.Cell.Red
    assert active_map[(1, 2)] == p.Cell.Green
    assert active_map[(2, 4)] == p.Cell.Blue
    assert active_map[(3, 6)] == p.Cell.Yellow
    assert active_map[(5, 12)] == p.Cell.Ojama

def test_actions():
    a = p.Action(p.ActionType.PUT, 2, p.Rotation.Right)
    assert a.type == p.ActionType.PUT
    assert a.x == 2
    assert a.rotation == p.Rotation.Right

def test_chain_erasure():
    """4連結の同色ぷよが Chain::execute により正常に消去されるか検証"""
    board = p.Board()
    # 0列目に赤ぷよを4つ縦に並べる
    board.set(0, 0, p.Cell.Red)
    board.set(0, 1, p.Cell.Red)
    board.set(0, 2, p.Cell.Red)
    board.set(0, 3, p.Cell.Red)

    erasure = p.Chain.execute(board)
    assert erasure.erased, "4連結の赤ぷよが消去対象として判定されていません"
    assert erasure.num_erased == 4, f"消去数が不一致です: expected 4, got {erasure.num_erased}"

def run_all():
    print("Running test_board_and_chain...")
    test_bitboard()
    test_board_and_gravity()
    test_get_active_puyos()
    test_actions()
    test_chain_erasure()
    print("  [PASS] test_board_and_chain")

if __name__ == "__main__":
    run_all()
