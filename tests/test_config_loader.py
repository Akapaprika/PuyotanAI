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
    assert hasattr(cfg, "dbs_max_similar")
    assert hasattr(cfg, "dbs_max_similar_end")
    assert hasattr(cfg, "dbs_empty_threshold_high")
    assert hasattr(cfg, "dbs_empty_threshold_low")
    assert hasattr(cfg, "dbs_auto_fill")
    assert hasattr(cfg, "get_dbs_limit_by_empty")
    assert cfg.dbs_auto_fill is True
    assert cfg.dbs_empty_threshold_high == 55
    assert cfg.dbs_empty_threshold_low == 18
    # 空きマス連動（78マス基準）のテスト: 55以上で3、18以下で8
    assert cfg.get_dbs_limit_by_empty(78) == 3
    assert cfg.get_dbs_limit_by_empty(55) == 3
    assert cfg.get_dbs_limit_by_empty(18) == 8
    assert cfg.get_dbs_limit_by_empty(0) == 8
    # 中間値の線形性チェック (18〜55の間で単調増加: 空きマス減少に伴い緩和)
    for e in range(54, 18, -1):
        assert cfg.get_dbs_limit_by_empty(e) >= cfg.get_dbs_limit_by_empty(e + 1)

def test_load_vs_config():
    cfg = p.load_vs_config(_CONFIG_PATH)
    assert cfg.beam_width > 0
    assert cfg.look_ahead > 0

def test_pure_search_apis():
    match = p.PuyotanMatch(12345)
    match.start()
    player = match.getPlayer(0)
    tsumo = match.getTsumo()

    solo_cfg = p.SoloBeamConfig()
    solo_cfg.beam_width = 100
    solo_cfg.look_ahead = 2
    act_idx, score = p.solo_beam_search(player, tsumo, solo_cfg)
    assert 0 <= act_idx < p.kNumRLActions

    vs_cfg = p.VsBeamConfig()
    vs_cfg.beam_width = 100
    vs_cfg.look_ahead = 2
    act_idx_vs, score_vs = p.vs_beam_search(player, tsumo, vs_cfg)
    assert 0 <= act_idx_vs < p.kNumRLActions

def run_all():
    print("Running test_config_loader...")
    test_load_solo_config()
    test_load_vs_config()
    test_pure_search_apis()
    print("  [PASS] test_config_loader")

if __name__ == "__main__":
    run_all()
