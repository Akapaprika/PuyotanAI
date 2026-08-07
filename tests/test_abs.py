"""
tests/test_abs.py
Adversarial Beam Search (ABS) の実行単体テスト。
"""
import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def main():
    print("Testing ABS module import...")
    cfg_path = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
    abs_cfg = p.load_abs_config(cfg_path)
    abs_cfg.depth = 2  # 軽量探索に設定し高速テスト
    
    match = p.PuyotanMatch(100)
    match.start()
    
    res = p.abs_search(match, 0, abs_cfg)
    assert 0 <= res.best_action < 22
    print(f"ABS Search Success! Best Action: {res.best_action}, Best Eval: {res.best_eval}")

if __name__ == "__main__":
    main()
