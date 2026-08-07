"""
tests/test_config_loader.py
BeamConfigLoader (JSON 設定ファイル読み込み・保存) の単体テスト。
"""
import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
_CONFIG_PATH = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")

if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def test_load_solo_config():
    cfg = p.load_solo_config(_CONFIG_PATH)
    assert cfg.beam_width > 0
    assert cfg.look_ahead > 0
    assert hasattr(cfg, "full_beam_depth")
    assert hasattr(cfg, "min_beam_width_ratio")

def test_load_vs_config():
    cfg = p.load_vs_config(_CONFIG_PATH)
    assert cfg.beam_width > 0
    assert cfg.look_ahead > 0

def run_all():
    print("Running test_config_loader...")
    test_load_solo_config()
    test_load_vs_config()
    print("  [PASS] test_config_loader")

if __name__ == "__main__":
    run_all()
