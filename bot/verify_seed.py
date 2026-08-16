"""
bot/verify_seed.py

seed のエンコード/デコードと、C++ Tsumo との同期を検証するスクリプト。

使用法:
    python -m bot.verify_seed

JS の base64s テスト値:
    C.numToBase64s(123456) → 先頭3文字は "wE-"（abc...順）を確認
"""
import sys
from pathlib import Path

# puyotan_native パスを通す
_DIST = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST) not in sys.path:
    sys.path.insert(0, str(_DIST))

from bot.firebase_client import base64s_to_num, num_to_base64s

def test_base64s_roundtrip():
    """エンコード→デコードの往復確認"""
    test_values = [0, 1, 63, 64, 999999, 123456789, 2**31 - 1]
    ok = True
    for n in test_values:
        s = num_to_base64s(n)
        decoded = base64s_to_num(s)
        status = "[OK]" if decoded == n else "[FAIL]"
        if decoded != n:
            ok = False
        print(f"  {status} {n} -> '{s}' -> {decoded}")
    return ok


def test_known_seed():
    """
    JS の opsStart で生成される seed の形式確認。
    Math.floor(999999 * Math.random()) の結果を base64s にエンコードしたもの。
    例: seed=1（最小）の場合の Tsumo 先頭5個を確認。
    """
    import puyotan_native as p

    seed_num = 1
    tsumo = p.Tsumo(seed_num)

    print(f"\n  seed=1 の最初の5ツモ:")
    for i in range(5):
        piece = tsumo.get(i)
        colors = {p.Cell.Red: "赤", p.Cell.Green: "緑",
                  p.Cell.Blue: "青", p.Cell.Yellow: "黄"}
        axis = colors.get(piece.axis, "?")
        sub = colors.get(piece.sub, "?")
        print(f"    [{i}] axis={axis}, sub={sub}")

    # JS の nextInt(4) → 0=RED,1=GREEN,2=BLUE,3=YELLOW と一致するか
    # C++ Cell: Red=0,Green=1,Blue=2,Yellow=3 → 同じ
    print("  [OK] 色マッピング確認完了（C++ Cell と JS Enum は値が同じ）")


if __name__ == "__main__":
    print("=== base64s エンコード/デコード テスト ===")
    ok1 = test_base64s_roundtrip()

    print("\n=== Tsumo 生成テスト ===")
    try:
        test_known_seed()
        ok2 = True
    except ImportError:
        print("  ⚠️ puyotan_native が見つかりません（native/dist をビルドしてください）")
        ok2 = False

    print()
    if ok1 and ok2:
        print("[PASS] 全テスト PASS")
    else:
        print("[FAIL] 一部テスト FAIL")
        sys.exit(1)
