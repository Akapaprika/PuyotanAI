"""
bot/bot_agent.py

ぷよたんβ連携 Bot のメインループ。

【動作の仕組み】
  Bot は Firestore に直接アクセスすることでサイトと連携します。

  1. 着席: /rooms/{roomId}/users/{slot} に {uid, name} を書き込む
     → サイト上で「着席中」として表示され、ゴーストも正しく表示される

  2. ゲーム開始 (both モードのみ):
     両席が Bot で埋まったら自動で /games に新ゲームを作成し
     /rooms/{roomId}.gameId を更新する

  3. 手を送信: /games/{gameId}/players/{slot}/actionMap に手を書き込む
     → サイトの Firestore リスナーが検知して reflectAction() が走り盤面が進む

  4. 退席: Bot 停止時 (Ctrl+C) に /rooms/{roomId}/users/{slot} を削除する

着席モード:
  - 1P のみ (bot_players={0}): 1P席に着席。2P は人間がサイトで着席し開始。
  - 2P のみ (bot_players={1}): 2P席に着席。1P は人間がサイトで着席し開始。
  - 両方   (bot_players={0,1}): 両席に着席し自動でゲームを開始する（ソロ）。
"""
from __future__ import annotations

import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

from bot.firebase_client import FirebaseClient
from bot.game_sync import GameState, cpp_action_to_js
from bot.firebase_client import num_to_base64s

_CONFIG_PATH = str(
    Path(__file__).parent.parent / "native" / "resources" / "beam_config.json"
)


class PuyotanBot:
    """
    ぷよたんβ サイト連携 Bot。

    Parameters
    ----------
    client      : FirebaseClient  — Firestore クライアント
    room_id     : str             — 参加するルームID（例: "e"）
    bot_players : set[int]        — Botが担当するプレイヤーID
                                    {0}=1Pのみ, {1}=2Pのみ, {0,1}=両方(ソロ)
    bot_name    : str             — サイト上で表示するBot名
    beam_width  : int, optional   — ビームサーチ幅（-1 でデフォルト）
    look_ahead  : int, optional   — 先読み手数（-1 でデフォルト）
    verbose     : bool            — 詳細ログを出力するかどうか
    """

    def __init__(
        self,
        client: FirebaseClient,
        room_id: str,
        bot_players: set[int],
        bot_name: str = "藍パプリカ",
        beam_width: int = -1,
        look_ahead: int = -1,
        verbose: bool = True,
    ) -> None:
        self.client = client
        self.room_id = room_id
        self.bot_players = bot_players
        self.bot_name = bot_name
        self.beam_width = beam_width
        self.look_ahead = look_ahead
        self.verbose = verbose
        self.is_solo = (bot_players == {0, 1})

        # Bot 固有の UID（サイト上でゴースト表示の識別に使用）
        self._bot_uid = f"bot-{uuid.uuid4().hex[:12]}"

        self._game_id: Optional[str] = None
        self._state: Optional[GameState] = None
        self._lock = threading.Lock()

        self._cancel_room: Optional[callable] = None
        self._cancel_players: list[Optional[callable]] = [None, None]

        # 送信済みフレームの追跡（プレイヤーIDごと）
        self._submitted: dict[int, set[int]] = {0: set(), 1: set()}

        # ビームサーチの非同期実行管理（プレイヤーIDごと）
        self._search_threads: dict[int, Optional[threading.Thread]] = {0: None, 1: None}
        self._search_results: dict[int, Optional[tuple[int, float]]] = {0: None, 1: None}
        self._search_lock = threading.Lock()

        # 頂点配列でのコントロール用
        self._should_stop = False

        # 重複ログ防止
        self._last_logged_state: str = ""

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        """部屋への着席・監視を開始する（ブロッキング）。"""
        self._log(f"Bot起動: room={self.room_id}, mode={self._mode_label()}")
        self._log(f"Bot UID: {self._bot_uid}")

        # 着席
        self._join_seats()

        # ルーム監視開始
        self._cancel_room = self.client.observe_room(
            self.room_id, self._on_room_update
        )
        self._log("Firestore ルーム監視開始...")

        # ソロモードならゲームを自動で開始
        if self.is_solo:
            time.sleep(0.8)
            self._start_solo_game()

        try:
            while not self._should_stop:
                time.sleep(0.3)
                self._tick()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """リスナー停止・退席処理。"""
        if self._cancel_room:
            self._cancel_room()
        for cancel in self._cancel_players:
            if cancel:
                cancel()
        self._leave_seats()
        self.client.close()
        self._log("退席・停止完了")

    def _mode_label(self) -> str:
        if self.bot_players == {0}:
            return "1P Bot (P2は人間)"
        elif self.bot_players == {1}:
            return "2P Bot (P1は人間)"
        else:
            return "Both / Solo (1P・2PともにBot)"

    # ------------------------------------------------------------------
    # 着席・退席
    # ------------------------------------------------------------------

    def _join_seats(self) -> None:
        """
        Bot起動時に前回セッションの残留データをリセットし、Botが担当する席に着席する。

        リセットにより以下がクリアされる:
          - /rooms/{roomId}/users/0  （1P席が古いBot UIDのままなら、人間が落下ボタンを押せない）
          - /rooms/{roomId}/users/1  （2P席同様）
          - /rooms/{roomId}.gameId   （古いgameIdが残るとゲーム画面が凖まる）
        """
        self._log("前回セッションのデータをリセット中...")
        try:
            self.client.abort_game(self.room_id)
        except Exception as e:
            self._log(f"[WARN] リセット中にエラー: {e}")

        # 着席
        for pid in sorted(self.bot_players):
            self._log(f"P{pid+1}席に着席: name='{self.bot_name}'")
            try:
                self.client.join_room(
                    self.room_id, pid, self._bot_uid, self.bot_name
                )
            except Exception as e:
                self._log(f"[WARN] P{pid+1} 着席失敗: {e}")

    def _leave_seats(self) -> None:
        """Bot が担当する席から退席する。"""
        for pid in sorted(self.bot_players):
            try:
                self.client.leave_room(self.room_id, pid)
                self._log(f"P{pid+1}席から退席しました")
            except Exception as e:
                self._log(f"[WARN] P{pid+1} 退席失敗: {e}")

    # ------------------------------------------------------------------
    # ゲーム開始（ソロモード）
    # ------------------------------------------------------------------

    def _start_solo_game(self) -> None:
        """
        ソロモード（両席がBot）のときにゲームを自動開始する。
        ランダムな seed を生成して /games に新ゲームを作成する。
        """
        seed = random.randint(0, 999999)
        seed_str = num_to_base64s(seed)
        self._log(f"ゲーム自動開始: seed='{seed_str}' ({seed})")
        try:
            game_id = self.client.new_game(self.room_id, seed_str)
            self._log(f"ゲーム開始完了: gameId={game_id}")
        except Exception as e:
            self._log(f"[ERROR] ゲーム開始失敗: {e}")

    # ------------------------------------------------------------------
    # ルーム更新ハンドラ
    # ------------------------------------------------------------------

    def _on_room_update(self, room_data: dict) -> None:
        """Firestoreのルームリスナーから呼ばれる（バックグラウンドスレッド）。"""
        new_game_id = room_data.get("gameId")
        with self._lock:
            if new_game_id == self._game_id:
                return
            self._game_id = new_game_id
            self._state = None
            self._submitted = {0: set(), 1: set()}

            for cancel in self._cancel_players:
                if cancel:
                    cancel()
            self._cancel_players = [None, None]

            if new_game_id:
                self._log(f"ゲーム検知: {new_game_id}")
            else:
                self._log("ゲーム終了（gameId=null）")
                # ソロモードなら次のゲームを自動開始
                if self.is_solo:
                    threading.Thread(
                        target=self._restart_solo_game, daemon=True
                    ).start()
                return

        # 別スレッドでゲームセットアップ
        threading.Thread(
            target=self._setup_game, args=(new_game_id,), daemon=True
        ).start()

    def _restart_solo_game(self) -> None:
        """ソロモードでゲーム終了後に次のゲームを自動開始する。"""
        time.sleep(1.5)  # 少し待つ（ゲーム終了のFirestore反映を待つ）
        with self._lock:
            if self._game_id is not None:
                return  # 既に次のゲームが始まっていたら不要
        self._log("次のゲームを自動開始します...")
        self._start_solo_game()

    def _setup_game(self, game_id: str) -> None:
        """ゲームドキュメントを取得して GameState を初期化する。"""
        time.sleep(0.5)  # Firestoreへの書き込み完了を少し待つ

        with self._lock:
            if self._game_id != game_id:
                return

        game_doc = self.client.fetch_game(game_id)
        if not game_doc:
            self._log(f"[ERROR] game doc not found: {game_id}")
            return

        seed_str = game_doc.get("seed", "")
        self._log(f"seed='{seed_str}'")

        maps = [
            self.client.fetch_game_player(game_id, 0) or {},
            self.client.fetch_game_player(game_id, 1) or {},
        ]

        with self._lock:
            if self._game_id != game_id:
                return
            state = GameState(seed_str, self.bot_players)
            for pid in [0, 1]:
                am = maps[pid].get("actionMap", {})
                if am:
                    state.update_action_map(pid, am)
            self._state = state
            self._log(
                f"GameState初期化: frame={state.current_frame},"
                f" playing={state.is_playing}"
            )

        # プレイヤーアクション監視を開始
        for pid in [0, 1]:
            cancel = self.client.observe_game_player(
                game_id, pid, self._make_player_handler(pid, game_id)
            )
            with self._lock:
                if self._game_id != game_id:
                    cancel()
                    return
                self._cancel_players[pid] = cancel

    def _make_player_handler(self, player_id: int, game_id: str):
        def handler(data: Optional[dict]):
            if data is None:
                return
            am = data.get("actionMap", {})
            if not am:
                return
            with self._lock:
                if self._state is None or self._game_id != game_id:
                    return
                old_frame = self._state.current_frame
                self._state.update_action_map(player_id, am)
                new_frame = self._state.current_frame
                if new_frame != old_frame:
                    self._log(
                        f"P{player_id+1} 更新:"
                        f" frame {old_frame} -> {new_frame}"
                    )
        return handler

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """0.3秒ごとに呼ばれる。各Botプレイヤーの手を処理する。"""
        # 探索結果の回収
        with self._search_lock:
            for pid in list(self.bot_players):
                if self._search_results[pid] is not None:
                    action_idx, score = self._search_results[pid]
                    self._search_results[pid] = None
                    self._search_threads[pid] = None
                    action = p.get_rl_action(action_idx)
                    self._submit_action(pid, action, score)

        with self._lock:
            state = self._state
            game_id = self._game_id

        if state is None or game_id is None:
            return

        # ゲーム終了検知
        if not state.is_playing:
            self._log("ゲーム終了。Bot を停止します。")
            self._should_stop = True
            return

        self._log_state(state)

        for pid in self.bot_players:
            self._try_start_search(state, pid)

    def _log_state(self, state: GameState) -> None:
        """状態変化があった時だけログを出す。"""
        frame = state.current_frame
        mask = state.match.getDecisionMask()
        key = f"{frame},{mask}"
        if key == self._last_logged_state:
            return
        self._last_logged_state = key
        dir_names = ["上", "右", "下", "左"]
        parts = []
        for pid in [0, 1]:
            needs = bool(mask & (1 << pid))
            done = frame in self._submitted[pid]
            if done:
                a = state.action_maps[pid].get(frame)
                if a:
                    parts.append(
                        f"P{pid+1}:送信済({a['x']}列{dir_names[a['dir']]})"
                    )
                else:
                    parts.append(f"P{pid+1}:送信済")
            elif needs:
                parts.append(f"P{pid+1}:思考中")
            else:
                parts.append(f"P{pid+1}:待機")
        self._log(f"frame={frame} | {' | '.join(parts)}")

    def _try_start_search(self, state: GameState, pid: int) -> None:
        if not state.needs_action(pid):
            return
        frame = state.current_frame
        if frame in self._submitted[pid]:
            return

        with self._search_lock:
            if (self._search_threads[pid] is not None
                    and self._search_threads[pid].is_alive()):
                return

        # ソロモード最適化: P0 の探索結果を P1 にも使い回す
        # (両者のフィールドは常に同一のため同じ最善手になる)
        if self.is_solo and pid == 1:
            with self._search_lock:
                if self._search_results[0] is not None:
                    # P0 の結果がすでにある場合はそれを P1 にも使う
                    self._search_results[1] = self._search_results[0]
                    return
                # P0 の探索スレッドが生きていれば P1 は待つ
                if (self._search_threads[0] is not None
                        and self._search_threads[0].is_alive()):
                    return

        player = state.get_player_state(pid)
        tsumo = state.get_tsumo()
        is_solo = self.is_solo

        def worker(p_id=pid, pl=player, t=tsumo):
            result = p.beam_search_action(
                pl, t, _CONFIG_PATH,
                self.beam_width, self.look_ahead,
                is_solo, False,
            )
            with self._search_lock:
                self._search_results[p_id] = result

        with self._search_lock:
            th = threading.Thread(target=worker, daemon=True)
            self._search_threads[pid] = th
            th.start()

    def _submit_action(self, pid: int, action: p.Action, score: float) -> None:
        with self._lock:
            state = self._state
            game_id = self._game_id

        if state is None or game_id is None:
            return
        if not state.is_playing or not state.needs_action(pid):
            return

        frame = state.current_frame
        if frame in self._submitted[pid]:
            return

        js_action = cpp_action_to_js(action)
        dir_names = ["上", "右", "下", "左"]
        self._log(
            f"P{pid+1} 送信: frame={frame},"
            f" {js_action['x']}列目 {dir_names[js_action['dir']]}向き,"
            f" score={score:.0f}"
        )

        existing = dict(state.action_maps[pid])
        existing[frame] = js_action

        try:
            self.client.send_action_map(game_id, pid, existing)
            self._submitted[pid].add(frame)
        except Exception as e:
            self._log(f"[ERROR] P{pid+1} 送信失敗: {e}")

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            line = f"[Bot] {msg}\n"
            try:
                sys.stdout.buffer.write(line.encode("utf-8"))
                sys.stdout.buffer.flush()
            except Exception:
                print(line, end="", flush=True)
