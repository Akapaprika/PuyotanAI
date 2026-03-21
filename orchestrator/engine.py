"""
C++ エンジン高レベルラッパー（PuyotanMatch ベース）
旧バージョンの Simulator ラッパーを廃止し、PuyotanMatch を使用するように刷新。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR / "native" / "dist", BASE_DIR / "native" / "build_Release" / "Release"]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

import puyotan_native as p


class PuyotanEngine:
    """
    PuyotanMatch の高レベル Python ラッパー。
    GUI / テスト / orchestrator から使用する。
    """

    def __init__(self, seed: int = 0):
        self.match = p.PuyotanMatch(seed)

    def start(self):
        self.match.start()
        self.match.step_until_decision()

    def is_playing(self) -> bool:
        return self.match.status == p.MatchStatus.PLAYING

    @property
    def status(self):
        return self.match.status

    @property
    def frame(self) -> int:
        return self.match.frame

    def get_player(self, pid: int):
        return self.match.getPlayer(pid)

    def get_piece(self, pid: int):
        return self.match.getPiece(pid)

    def set_action(self, pid: int, action) -> bool:
        return self.match.setAction(pid, action)

    def step_until_decision(self):
        return self.match.step_until_decision()
