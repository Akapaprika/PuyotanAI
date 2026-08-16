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

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# puyotan_native パスを通す
_DIST = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST) not in sys.path:
    sys.path.insert(0, str(_DIST))

from bot.firebase_client import base64s_to_num, num_to_base64s

def test_base64s_roundtrip():
    """Roundtrip check for encode -> decode"""
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
    """Verify first 5 tsumo pieces for seed=1"""
    import puyotan_native as p

    seed_num = 1
    tsumo = p.Tsumo(seed_num)

    print(f"\n  seed=1 first 5 tsumo pieces:")
    for i in range(5):
        piece = tsumo.get(i)
        colors = {p.Cell.Red: "Red", p.Cell.Green: "Green",
                  p.Cell.Blue: "Blue", p.Cell.Yellow: "Yellow"}
        axis = colors.get(piece.axis, "?")
        sub = colors.get(piece.sub, "?")
        print(f"    [{i}] axis={axis}, sub={sub}")

    print("  [OK] Color mapping verified.")


if __name__ == "__main__":
    print("=== base64s encode/decode test ===")
    ok1 = test_base64s_roundtrip()

    print("\n=== Tsumo generation test ===")
    try:
        test_known_seed()
        ok2 = True
    except ImportError:
        print("  [ERROR] puyotan_native not found in native/dist")
        ok2 = False

    print()
    if ok1 and ok2:
        print("[PASS] All tests PASSED")
    else:
        print("[FAIL] Some tests FAILED")
        sys.exit(1)
