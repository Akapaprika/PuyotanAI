"""
bot/diagnose.py

Bot とサイトの連携状況を診断するスクリプト。
実際にどのゲームが進行中で、どのアクションが送信済みかを確認する。

使用法:
    python -m bot.diagnose --room e
"""
import sys
import argparse
from pathlib import Path

_DIST = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST) not in sys.path:
    sys.path.insert(0, str(_DIST))

import puyotan_native as p
from bot.firebase_client import FirebaseClient, base64s_to_num
from bot.game_sync import GameState, cpp_action_to_js

ROOMS = {
    "a": "対戦部屋（上級）A",
    "b": "対戦部屋（上級）B",
    "c": "対戦部屋（初級）A",
    "d": "対戦部屋（初級）D",
    "e": "身内部屋 A",
    "f": "身内部屋 B",
    "g": "身内部屋 C",
    "h": "身内部屋 D",
}

COLORS = {
    p.Cell.Red: "赤", p.Cell.Green: "緑",
    p.Cell.Blue: "青", p.Cell.Yellow: "黄", p.Cell.Ojama: "おじゃま",
}

DIR_NAMES = ["上", "右", "下", "左"]


def diagnose(room_id: str) -> None:
    print("=" * 55)
    print(f"  診断: room={room_id} ({ROOMS.get(room_id, '?')})")
    print("=" * 55)

    client = FirebaseClient()

    # -----------------------------------------------------------------------
    # 1. ルーム状態確認
    # -----------------------------------------------------------------------
    print("\n[1] ルーム状態")
    room = client.db.collection("rooms").document(room_id).get().to_dict()
    if not room:
        print("  ERR: ルームが見つかりません")
        return

    game_id = room.get("gameId")
    print(f"  gameId : {game_id or 'null (ゲーム未開始)'}")

    # 着席状況
    users_ref = (client.db.collection("rooms").document(room_id)
                 .collection("users").stream())
    users = {doc.id: doc.to_dict() for doc in users_ref}
    for slot in ["0", "1"]:
        u = users.get(slot)
        if u:
            print(f"  {int(slot)+1}P席  : 着席中 ({u.get('name', '?')})")
        else:
            print(f"  {int(slot)+1}P席  : 空席")

    if not game_id:
        print("\n  -> ゲームが開始されていません。")
        print("     サイトで両プレイヤーが着席し、'ゲーム開始'を押してください。")
        return

    # -----------------------------------------------------------------------
    # 2. ゲーム状態確認
    # -----------------------------------------------------------------------
    print(f"\n[2] ゲーム状態 ({game_id})")
    game = client.fetch_game(game_id)
    if not game:
        print("  ERR: ゲームドキュメントが見つかりません")
        return

    seed_str = game.get("seed", "")
    seed_num = base64s_to_num(seed_str)
    print(f"  seed   : '{seed_str}' ({seed_num})")

    # -----------------------------------------------------------------------
    # 3. 各プレイヤーのアクション確認
    # -----------------------------------------------------------------------
    print(f"\n[3] アクション状況")
    action_maps = {}
    for pid in [0, 1]:
        doc = client.fetch_game_player(game_id, pid)
        am = {}
        if doc and doc.get("actionMap"):
            am = {int(k): v for k, v in doc["actionMap"].items()}
        action_maps[pid] = am
        frames_list = sorted(am.keys())
        print(f"  P{pid+1}: {len(am)} フレーム分送信済み"
              + (f" (最後: frame {frames_list[-1]})" if frames_list else ""))

    # -----------------------------------------------------------------------
    # 4. 盤面復元・現在フレーム確認
    # -----------------------------------------------------------------------
    print(f"\n[4] 盤面復元")
    try:
        state = GameState(seed_str, {0, 1})
        for pid in [0, 1]:
            if action_maps[pid]:
                state.update_action_map(pid, {str(k): v for k, v in action_maps[pid].items()})

        frame = state.current_frame
        print(f"  現在フレーム : {frame}")
        print(f"  ゲーム進行中 : {state.is_playing}")

        mask = state.match.getDecisionMask()
        for pid in [0, 1]:
            needs = bool(mask & (1 << pid))
            submitted = frame in action_maps[pid]
            if submitted:
                a = action_maps[pid][frame]
                status = f"送信済み (x={a['x']}, {DIR_NAMES[a['dir']]})"
            elif needs:
                status = "未送信（Bot/プレイヤーが入力待ち）"
            else:
                status = "入力不要（連鎖/落下中など）"
            print(f"  P{pid+1} frame{frame}: {status}")

        # -----------------------------------------------------------------------
        # 5. ツモ確認
        # -----------------------------------------------------------------------
        if state.is_playing:
            print(f"\n[5] 現在ツモ（P1視点）")
            tsumo = state.get_tsumo()
            p0 = state.get_player_state(0)
            for i in range(3):
                piece = tsumo.get(p0.active_next_pos + i)
                a = COLORS.get(piece.axis, "?")
                s = COLORS.get(piece.sub, "?")
                marker = "<-- 現在" if i == 0 else ""
                print(f"  [{i}] {a}/{s} {marker}")

        # -----------------------------------------------------------------------
        # 6. 診断まとめ
        # -----------------------------------------------------------------------
        print(f"\n[診断まとめ]")
        p0_submitted = frame in action_maps[0]
        p1_submitted = frame in action_maps[1]

        if not state.is_playing:
            print("  ゲームが終了しています。サイトで新しいゲームを開始してください。")
        elif p0_submitted and p1_submitted:
            print("  両プレイヤーとも送信済みですが盤面が進んでいません。")
            print("  -> サイトのページをリロードして確認してください。")
        elif p0_submitted and not p1_submitted:
            print("  P1はBot送信済み。P2の入力待ちです。")
            print("  -> サイトでP2の人間プレイヤーが手を選んでください。")
        elif not p0_submitted and p1_submitted:
            print("  P2はBot/人間送信済み。P1の入力待ちです。")
        else:
            print("  両プレイヤーとも未送信。Botが動作しているか確認してください。")

    except Exception as e:
        import traceback
        print(f"  ERR: 盤面復元中にエラーが発生しました: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot連携状況診断")
    parser.add_argument("--room", "-r", default="e",
                        help="部屋ID (例: e)")
    args = parser.parse_args()
    diagnose(args.room)
