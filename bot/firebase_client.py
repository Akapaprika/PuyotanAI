"""
bot/firebase_client.py

Firebase Firestore へのアクセスをラップするモジュール。
google.cloud.firestore を AnonymousCredentials で使用するため、
サービスアカウントキー不要で動作する。

Firestore スキーマ:
  /rooms/{roomId}                          → {gameId, name}
  /rooms/{roomId}/users/{0 or 1}           → {uid, name}  ← 着席情報
  /games/{gameId}                          → {seed, startAt}
  /games/{gameId}/players/{0 or 1}         → {actionMap}  ← Botが書き込む
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore as gf


class FirebaseClient:
    """
    Firestore の読み書きを担うクライアント。
    サービスアカウント不要・匿名アクセスで動作する。
    """

    PROJECT_ID = "puyotan-be458"

    def __init__(self) -> None:
        self.db = gf.Client(
            project=self.PROJECT_ID,
            credentials=AnonymousCredentials(),
        )
        self._listeners: list = []

    # ------------------------------------------------------------------
    # ルーム監視
    # ------------------------------------------------------------------
    def observe_room(
        self, room_id: str, callback: Callable[[dict], None]
    ) -> Callable[[], None]:
        """
        /rooms/{roomId} を監視し、変更があるたびに callback を呼ぶ。
        Returns: 監視を停止する関数
        """
        doc_ref = self.db.collection("rooms").document(room_id)

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                data = doc.to_dict()
                if data:
                    callback(data)

        watch = doc_ref.on_snapshot(on_snapshot)
        self._listeners.append(watch)
        return watch.unsubscribe

    # ------------------------------------------------------------------
    # ゲームドキュメント取得
    # ------------------------------------------------------------------
    def fetch_game(self, game_id: str) -> Optional[dict]:
        """
        /games/{gameId} を取得して辞書で返す。
        """
        doc = self.db.collection("games").document(game_id).get()
        return doc.to_dict() if doc.exists else None

    # ------------------------------------------------------------------
    # プレイヤーアクション監視
    # ------------------------------------------------------------------
    def observe_game_player(
        self,
        game_id: str,
        player_id: int,
        callback: Callable[[Optional[dict]], None],
    ) -> Callable[[], None]:
        """
        /games/{gameId}/players/{playerId} を監視する。
        Returns: 監視を停止する関数
        """
        doc_ref = (
            self.db.collection("games")
            .document(game_id)
            .collection("players")
            .document(str(player_id))
        )

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                data = doc.to_dict() if doc.exists else None
                callback(data)

        watch = doc_ref.on_snapshot(on_snapshot)
        self._listeners.append(watch)
        return watch.unsubscribe

    def fetch_game_player(self, game_id: str, player_id: int) -> Optional[dict]:
        """
        /games/{gameId}/players/{playerId} を一度だけ取得する。
        """
        doc = (
            self.db.collection("games")
            .document(game_id)
            .collection("players")
            .document(str(player_id))
            .get()
        )
        return doc.to_dict() if doc.exists else None

    # ------------------------------------------------------------------
    # アクション送信
    # ------------------------------------------------------------------
    def send_action_map(
        self, game_id: str, player_id: int, action_map: dict[int, dict]
    ) -> None:
        """
        /games/{gameId}/players/{playerId} に actionMap を書き込む。

        Parameters
        ----------
        action_map : {frame_int: {"x": int, "dir": int}}
        """
        # Firestore のキーは文字列のみ
        serialized = {str(k): v for k, v in action_map.items()}
        (
            self.db.collection("games")
            .document(game_id)
            .collection("players")
            .document(str(player_id))
            .set({"actionMap": serialized})
        )

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 着席・退席
    # ------------------------------------------------------------------

    def join_room(
        self, room_id: str, player_slot: int, uid: str, name: str = "Bot"
    ) -> None:
        """
        /rooms/{roomId}/users/{playerSlot} に着席情報を書き込む。
        サイト上で「着席中」として表示され、ゴースト表示が有効になる。

        Parameters
        ----------
        player_slot : 0=1P, 1=2P
        uid         : Bot固有のUID（ゴースト表示の判定に使用）
        name        : サイト上で表示するBot名
        """
        (
            self.db.collection("rooms")
            .document(room_id)
            .collection("users")
            .document(str(player_slot))
            .set({"uid": uid, "name": name})
        )

    def leave_room(self, room_id: str, player_slot: int) -> None:
        """退席（/rooms/{roomId}/users/{playerSlot} を削除）。"""
        (
            self.db.collection("rooms")
            .document(room_id)
            .collection("users")
            .document(str(player_slot))
            .delete()
        )

    # ------------------------------------------------------------------
    # ゲーム開始
    # ------------------------------------------------------------------

    def new_game(self, room_id: str, seed_str: str) -> str:
        """
        新しいゲームを開始する。
        /games に新しいドキュメントを作成し、/rooms/{roomId}.gameId を更新する。

        Parameters
        ----------
        seed_str : base64s形式のseed文字列

        Returns
        -------
        str : 新しい gameId
        """
        from google.cloud import firestore as gf

        # /games コレクションに新しいドキュメントを追加
        game_ref = self.db.collection("games").document()
        game_ref.set({
            "seed": seed_str,
            "startAt": gf.SERVER_TIMESTAMP,
        })

        # /rooms/{roomId}.gameId を更新
        self.db.collection("rooms").document(room_id).update(
            {"gameId": game_ref.id}
        )

        return game_ref.id

    def abort_game(self, room_id: str) -> None:
        """
        ゲームを強制終了する。
        両プレイヤーを退席させ、gameIdをnullに設定する。
        """
        for slot in [0, 1]:
            try:
                self.leave_room(room_id, slot)
            except Exception:
                pass
        self.db.collection("rooms").document(room_id).set(
            {"gameId": None}, merge=True
        )

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------

    def close(self) -> None:
        for watch in self._listeners:
            try:
                watch.unsubscribe()
            except Exception:
                pass
        self._listeners.clear()


# ------------------------------------------------------------------
# seed のエンコード/デコード（サイト独自の base64s 形式）
# ------------------------------------------------------------------
_BASE64_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
_CHAR_TO_IDX = {c: i for i, c in enumerate(_BASE64_CHARS)}


def base64s_to_num(s: str) -> int:
    """
    サイト独自の base64s 文字列を整数に変換する。
    JS の C.base64stoNum() と同等。
    """
    result = 0
    multiplier = 1
    for ch in s:
        result += _CHAR_TO_IDX.get(ch, 0) * multiplier
        multiplier *= 64
    return result


def num_to_base64s(n: int) -> str:
    """
    整数をサイト独自の base64s 文字列に変換する。
    JS の C.numToBase64s() と同等。
    """
    if n < 0:
        return "?"
    result = ""
    for _ in range(10):
        result += _BASE64_CHARS[n % 64]
        n //= 64
        if n == 0:
            break
    return result
