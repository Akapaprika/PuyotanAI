"""
bot/game_sync.py

Firestore の actionMap から C++ の PuyotanMatch を完全に復元するモジュール。

座標系の差異:
  JS の x は 1始まり (1〜6)、C++ の x は 0始まり (0〜5) → x_cpp = x_js - 1
  JS の dir と C++ の Rotation は同じ数値: Up=0, Right=1, Down=2, Left=3
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

from bot.firebase_client import base64s_to_num

# Rotation ↔ dir の対応テーブル
_ROTATION_TO_DIR = {
    p.Rotation.Up:    0,
    p.Rotation.Right: 1,
    p.Rotation.Down:  2,
    p.Rotation.Left:  3,
}
_DIR_TO_ROTATION = {v: k for k, v in _ROTATION_TO_DIR.items()}


# ------------------------------------------------------------------
# 手の変換
# ------------------------------------------------------------------

def js_action_to_cpp(x_js: int, dir_js: int) -> p.Action:
    """JS の {x, dir} を C++ の Action に変換する。"""
    return p.Action(
        p.ActionType.PUT,
        x_js - 1,                              # 1始まり → 0始まり
        _DIR_TO_ROTATION.get(dir_js, p.Rotation.Up),
    )


def cpp_action_to_js(action: p.Action) -> dict:
    """
    C++ の Action を JS の {x, dir, type} に変換する。

    type=1 は JS の G.PUT に相当。
    サイトの setAction() は switch(t.type) で判定するため必須。
    type がないと 'unsupported action type.' で例外が投げられ
    isActiveReflectAction が詰まりゲームが一切進行しなくなる。
    """
    return {
        "x":   action.x + 1,                   # 0始まり → 1始まり
        "dir": _ROTATION_TO_DIR.get(action.rotation, 0),
        "type": 1,                             # G.PUT = 1 (必須)
    }


# ------------------------------------------------------------------
# ゲーム状態の復元
# ------------------------------------------------------------------

class GameState:
    """
    Firestore の actionMap から PuyotanMatch を再構築・維持するクラス。

    Parameters
    ----------
    seed_str   : str         — base64s形式のseed文字列
    bot_players: set[int]    — Botが担当するプレイヤーID集合 {0}, {1}, {0,1}
    """

    def __init__(self, seed_str: str, bot_players: set[int]) -> None:
        self.seed_num = base64s_to_num(seed_str)
        self.bot_players = bot_players
        # action_maps[pid] = {frame_int: {"x": int, "dir": int}}
        self.action_maps: dict[int, dict[int, dict]] = {0: {}, 1: {}}
        self._rebuild_match()

    def _rebuild_match(self) -> None:
        """seed と現在の actionMap から PuyotanMatch を最初から再構築する。"""
        match = p.PuyotanMatch(self.seed_num)
        match.start()

        for frame in range(1, 1001):
            a0 = self.action_maps[0].get(frame)
            a1 = self.action_maps[1].get(frame)
            if a0 is not None:
                match.setAction(0, js_action_to_cpp(a0["x"], a0["dir"]))
            if a1 is not None:
                match.setAction(1, js_action_to_cpp(a1["x"], a1["dir"]))
            if not match.canStepNextFrame():
                break
            match.stepNextFrame()

        self.match = match

    def update_action_map(self, player_id: int, raw_action_map: dict) -> None:
        """Firestore から取得した actionMap で内部状態を更新し、match を再構築。"""
        self.action_maps[player_id] = {int(k): v for k, v in raw_action_map.items()}
        self._rebuild_match()

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> int:
        return self.match.frame

    @property
    def is_playing(self) -> bool:
        return self.match.status == p.MatchStatus.PLAYING

    def needs_action(self, player_id: int) -> bool:
        """指定プレイヤーが現フレームで手を入力すべきか。"""
        mask = self.match.getDecisionMask()
        return bool(mask & (1 << player_id))

    def is_solo_mode(self) -> bool:
        """両プレイヤーをBotが担当するソロモード判定。"""
        return self.bot_players == {0, 1}

    def already_submitted(self, player_id: int) -> bool:
        """指定プレイヤーの手がすでに現フレームに登録済みか。"""
        return self.current_frame in self.action_maps[player_id]

    def get_player_state(self, player_id: int) -> p.PuyotanPlayer:
        return self.match.getPlayer(player_id)

    def get_tsumo(self) -> p.Tsumo:
        return self.match.getTsumo()
