"""
bot/main.py

ぷよたんβ連携 Bot のエントリーポイント。

設定は bot/bot.json で管理できます。コマンド引数が指定された場合はそちらが優先されます。

使用例:
    # 設定ファイルの設定で起動（引数なし）
    python -m bot.main

    # 身内部屋Aで両方操作（ソロモード）
    python -m bot.main --room e --player both

    # 対戦部屋Aで1P席のみ（相手が来るまで待機、ゲーム開始後に動作）
    python -m bot.main --room a --player 1p --name "PuyotanAI"

    # Bot名・ビームサーチパラメータ指定
    python -m bot.main --room e --player both --name "強いBot" --beam-width 500 --look-ahead 10
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows regardless of the active code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bot.firebase_client import FirebaseClient
from bot.bot_agent import PuyotanBot

_DEFAULT_CONFIG = Path(__file__).parent / "bot.json"

_ROOM_DESCRIPTIONS = {
    "a": "対戦部屋（上級）A",
    "b": "対戦部屋（上級）B",
    "c": "対戦部屋（初級）A",
    "d": "対戦部屋（初級）B",
    "e": "身内部屋 A",
    "f": "身内部屋 B",
    "g": "身内部屋 C",
    "h": "身内部屋 D",
}


def parse_player_mode(value: str) -> set[int]:
    v = value.strip().lower()
    if v in ("1p", "1", "p1"):
        return {0}
    elif v in ("2p", "2", "p2"):
        return {1}
    elif v in ("both", "solo", "all", "b"):
        return {0, 1}
    else:
        raise ValueError(
            f"'{value}' は不正なプレイヤー指定です。"
            " '1p', '2p', 'both' のいずれかを指定してください。"
        )


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_comment")}
    except Exception as e:
        print(f"[Bot] 警告: 設定ファイル読み込み失敗 ({config_path}): {e}", file=sys.stderr)
        return {}


def print_banner(room: str, mode_label: str, bot_name: str,
                 config_path: Path, is_solo: bool) -> None:
    room_name = _ROOM_DESCRIPTIONS.get(room, room)
    search_mode = "ソロ (soloBeamSearch)" if is_solo else "対戦 (vsBeamSearch)"
    lines = [
        "=" * 65,
        "  puyotan.refpuyo.net 連携 AI Bot",
        "=" * 65,
        f"  設定ファイル : {config_path}",
        f"  部屋         : {room} ({room_name})",
        f"  モード       : {mode_label}",
        f"  AI探索       : {search_mode}",
        f"  Bot名        : {bot_name}",
        "=" * 65,
    ]
    for line in lines:
        try:
            sys.stdout.buffer.write((line + "\n").encode("utf-8"))
        except Exception:
            print(line)
    sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="puyotan.refpuyo.net 連携 AI Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="設定ファイルのパス（デフォルト: bot/bot.json）",
    )
    parser.add_argument(
        "--room", "-r",
        default=None,
        help="対戦部屋ID: a=上級A, b=上級B, c=初級A, d=初級B, e=身内A, f=身内B, g=身内C, h=身内D",
    )
    parser.add_argument(
        "--player", "-p",
        default=None,
        help="着席席: '1p'=1P席のみ  '2p'=2P席のみ  'both'=両方(ソロ)",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="サイト上で表示するBot名（デフォルト: PuyotanAI）",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="ビームサーチ幅（デフォルト: beam_config.json の設定値）",
    )
    parser.add_argument(
        "--look-ahead",
        type=int,
        default=None,
        help="先読み手数（デフォルト: beam_config.json の設定値）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=None,
        help="ログ出力を抑制する",
    )

    args = parser.parse_args()

    # 設定ファイル読み込み
    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG
    cfg = load_config(config_path)

    # コマンド引数 > 設定ファイル > デフォルト の優先順位でマージ
    room       = args.room       or cfg.get("room",       "e")
    player_str = args.player     or cfg.get("player",     "both")
    bot_name   = args.name       or cfg.get("name",       "藍パプリカ")
    beam_width = args.beam_width if args.beam_width is not None else cfg.get("beam_width", -1)
    look_ahead = args.look_ahead if args.look_ahead is not None else cfg.get("look_ahead", -1)
    quiet      = args.quiet      or cfg.get("quiet",      False)

    # プレイヤーモード解析
    try:
        bot_players = parse_player_mode(player_str)
    except ValueError as e:
        parser.error(str(e))
        return

    mode_label = PuyotanBot.mode_label(bot_players)

    is_solo = (bot_players == {0, 1})

    if not quiet:
        print_banner(room, mode_label, bot_name, config_path, is_solo)

    # Firestore 接続
    try:
        client = FirebaseClient()
        if not quiet:
            print("[Bot] Firestore 接続成功", flush=True)
    except Exception as e:
        print(f"[Bot] ERROR: Firestore 接続失敗: {e}", file=sys.stderr)
        sys.exit(1)

    bot = PuyotanBot(
        client=client,
        room_id=room,
        bot_players=bot_players,
        bot_name=bot_name,
        beam_width=beam_width,
        look_ahead=look_ahead,
        verbose=not quiet,
    )

    bot.start()


if __name__ == "__main__":
    main()
