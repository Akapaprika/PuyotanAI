"""
bot/bot_agent.py

ぷよたんβ連携 Bot のメインループ。

【動作の仕組み】
  Bot は Firestore に直接アクセスすることでサイトと連携します。

  1. 着席: /rooms/{roomId}/users/{slot} に {uid, name} を書き込む
     → サイト上で「着席中」として表示される

  2. ゲーム開始 (both モードのみ):
     /games に新ゲームを作成し /rooms/{roomId}.gameId を更新する

  3. 手を送信: /games/{gameId}/players/{slot}/actionMap に書き込む
     → サイトの Firestore リスナーが検知して reflectAction() が走り盤面が進む

  4. 退席: Bot 停止時に /rooms/{roomId}/users/{slot} を削除する

着席モード:
  - 1P (bot_players={0}): 1P席に着席。2P は人間がサイトで着席し開始。
  - 2P (bot_players={1}): 2P席に着席。1P は人間がサイトで着席し開始。
  - both (bot_players={0,1}): 両席に着席し自動でゲームを開始する。

ソロモード (both) の探索について:
  両プレイヤーの盤面は常に同一のため、探索は1回だけ実行し
  その結果を P0・P1 に同時に送信する。
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

from ai import BeamSearchAgent, VsBeamSearchAgent
from bot.firebase_client import FirebaseClient, num_to_base64s
from bot.game_sync import GameState, cpp_action_to_js, js_action_to_cpp


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

        # AI エージェントの初期化（ソロなら BeamSearchAgent、VSなら相手注視・反撃対応の VsBeamSearchAgent）
        bw = beam_width if beam_width > 0 else None
        la = look_ahead if look_ahead > 0 else None
        if self.is_solo:
            self._agents = {0: BeamSearchAgent(beam_width=bw, look_ahead=la)}
            self._agents[0].on_mode_updated(is_solo=True)
        else:
            self._agents = {
                pid: VsBeamSearchAgent(enable_attack_search=True, beam_width=bw, look_ahead=la)
                for pid in self.bot_players
            }

        self._should_stop = False
        self._last_logged_state: str = ""

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        """部屋への着席・監視を開始する（ブロッキング）。"""
        self._log(f"Bot起動: room={self.room_id}, mode={self._mode_label()}")
        self._log(f"Bot UID: {self._bot_uid}")

        self._join_seats()
        if self._should_stop:
            return

        self._cancel_room = self.client.observe_room(
            self.room_id, self._on_room_update
        )
        self._log("Firestore ルーム監視開始...")

        if self.is_solo:
            time.sleep(0.8)  # 着席がFirestoreに伝わるのを少し待つ
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
            return "1P Bot（P2は人間）"
        elif self.bot_players == {1}:
            return "2P Bot（P1は人間）"
        else:
            return "Both / Solo（1P・2PともにBot）"

    # ------------------------------------------------------------------
    # 着席・退席
    # ------------------------------------------------------------------

    def _join_seats(self) -> None:
        """
        Bot起動時に前回セッションのデータをリセットし着席する。

        安全チェック:
          - アクティブなゲーム中 (gameId != null) は起動拒否
          - Bot以外のユーザーが席占めしている場合は起動拒否
        """
        # --- 事前チェック ---
        self._log("ルームの状態を確認中...")
        try:
            room_data = self.client.fetch_room(self.room_id)
        except Exception as e:
            self._log(f"[WARN] ルーム取得失敗: {e}")
            room_data = None

        if room_data:
            existing_game_id = room_data.get("gameId")
            if existing_game_id:
                # すでにゲームIDがある場合、本当に進行中か確認する
                is_active = True
                try:
                    game_doc = self.client.fetch_game(existing_game_id)
                    if game_doc:
                        seed_str = game_doc.get("seed", "")
                        maps = [
                            self.client.fetch_game_player(existing_game_id, 0) or {},
                            self.client.fetch_game_player(existing_game_id, 1) or {},
                        ]
                        state = GameState(seed_str, self.bot_players)
                        for pid in [0, 1]:
                            am = maps[pid].get("actionMap", {})
                            if am:
                                state.update_action_map(pid, am)
                        if not state.is_playing:
                            is_active = False
                except Exception as e:
                    self._log(f"[WARN] 既存ゲームの状態確認中にエラー: {e}")

                if is_active:
                    self._log(
                        f"[警告] 部屋 '{self.room_id}' は現在対戦中です (gameId={existing_game_id})。"
                        f"\n         対戦終了後に再起動してください。"
                    )
                    self._should_stop = True
                    return

            users = room_data.get("users", {})
            for pid in self.bot_players:
                user = users.get(str(pid), {})
                uid = user.get("uid", "")
                if uid and not uid.startswith("bot-"):
                    self._log(
                        f"[警告] P{pid+1}席は他のユーザーが使用中です"
                        f" (name='{user.get('name', '?')}', uid={uid})。"
                        f"\n         その席が空くのを待ってから再起動してください。"
                    )
                    self._should_stop = True
                    return

        # --- リセット・着席 ---
        self._log("前回セッションのデータをリセット中...")
        for pid in sorted(self.bot_players):
            self._log(f"P{pid+1}席に着席: name='{self.bot_name}'")
            try:
                self.client.join_room(
                    self.room_id, pid, self._bot_uid, self.bot_name
                )
            except Exception as e:
                self._log(f"[WARN] P{pid+1} 着席失敗: {e}")

    def _leave_seats(self) -> None:
        """Bot が担当する席から退席する（自分が座っている場合のみ）。"""
        try:
            room_data = self.client.fetch_room(self.room_id)
            users = room_data.get("users", {}) if room_data else {}
        except Exception:
            users = {}

        for pid in sorted(self.bot_players):
            user = users.get(str(pid), {})
            current_uid = user.get("uid", "")
            if current_uid == self._bot_uid:
                try:
                    self.client.leave_room(self.room_id, pid)
                    self._log(f"P{pid+1}席から退席しました")
                except Exception as e:
                    self._log(f"[WARN] P{pid+1} 退席失敗: {e}")
            else:
                self._log(f"P{pid+1}席は自分が着席していないため、退席処理をスキップします (現UID={current_uid})")

    # ------------------------------------------------------------------
    # ゲーム開始（ソロモード）
    # ------------------------------------------------------------------

    def _start_solo_game(self) -> None:
        """
        ソロモードでゲームを自動開始する。
        サイトの opsStart() と完全に同等:
          newGame()  → /games に新ゲームを作成
          sendChat() → ルームチャットに「ゲーム開始」を通知
        """
        seed = random.randint(0, 999999)
        seed_str = num_to_base64s(seed)
        self._log(f"ゲーム自動開始: seed='{seed_str}' ({seed})")
        try:
            game_id = self.client.new_game(self.room_id, seed_str)
            self.client.send_chat(
                self.room_id,
                text=f"ゲーム開始 {self.bot_name} vs {self.bot_name}",
                color="#0000ff",
                uid=self._bot_uid,
                name=self.bot_name,
            )
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

            old_game_id = self._game_id
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
                if old_game_id is not None:
                    self._log("ゲーム終了（強制終了またはゲームオーバー）。Bot を停止します。")
                    self._should_stop = True
                return

        threading.Thread(
            target=self._setup_game, args=(new_game_id,), daemon=True
        ).start()

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
                    self._log(f"P{player_id+1} 更新: frame {old_frame} -> {new_frame}")
        return handler

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """
        0.3秒ごとに呼ばれる。探索結果の回収・送信と新規探索の起動を行う。
        """
        with self._lock:
            state = self._state
            game_id = self._game_id

        if state is None or game_id is None:
            return

        # ゲーム終了検知
        if not state.is_playing:
            self._log("ゲーム終了。Bot を停止します。")
            try:
                self.client.send_chat(
                    self.room_id,
                    text="ゲームが終了しました",
                    color="#0000ff",
                    uid=self._bot_uid,
                    name=self.bot_name,
                    game_id=game_id,
                )
            except Exception:
                pass
            self._should_stop = True
            return

        self._log_state(state)
        self._process_ai_turns(state)

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
            done = frame in self._submitted[pid]
            if done:
                a = state.action_maps[pid].get(frame)
                label = f"P{pid+1}:送信済({a['x']}列{dir_names[a['dir']]})" if a else f"P{pid+1}:送信済"
                parts.append(label)
            elif bool(mask & (1 << pid)):
                parts.append(f"P{pid+1}:思考中")
            else:
                parts.append(f"P{pid+1}:待機")
        self._log(f"frame={frame} | {' | '.join(parts)}")

    def _process_ai_turns(self, state: GameState) -> None:
        """担当プレイヤーの AI 思考を進行させ、手が確定したら送信する。"""
        # ソロ（both）モード時の書き込みズレ（Syncズレ）修復
        if self.is_solo:
            frame = state.current_frame
            a0 = state.action_maps[0].get(frame)
            a1 = state.action_maps[1].get(frame)
            if a0 is not None and a1 is None:
                self._log(f"[Sync] P2の手をP1 ({a0['x']}列) からコピーして再送信します")
                self._submit_action(1, js_action_to_cpp(a0["x"], a0["dir"]), 0.0)
                return
            elif a1 is not None and a0 is None:
                self._log(f"[Sync] P1の手をP2 ({a1['x']}列) からコピーして再送信します")
                self._submit_action(0, js_action_to_cpp(a1["x"], a1["dir"]), 0.0)
                return

            # ソロモード: P0 エージェントで1回だけ思考し、両席に同時送信
            if state.needs_action(0) and (state.current_frame not in self._submitted[0]):
                agent = self._agents[0]
                action = agent.get_action(state, 0)
                if action is not None:
                    score = agent.last_score
                    self._submit_action(0, action, score)
                    self._submit_action(1, action, score)
        else:
            # VSモード: 担当プレイヤーごとに思考・送信
            for pid in sorted(self.bot_players):
                if state.needs_action(pid) and (state.current_frame not in self._submitted[pid]):
                    agent = self._agents[pid]
                    action = agent.get_action(state, pid)
                    if action is not None:
                        score = agent.last_score
                        self._submit_action(pid, action, score)

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
