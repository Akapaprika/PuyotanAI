"""
Puyotan Room Cleanup Script.
Resets the game status and leaves seats in the specified room.
"""
import argparse
from bot.firebase_client import FirebaseClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Puyotan Room Cleanup Script")
    parser.add_argument(
        "--room", "-r",
        default="e",
        help="Room ID to clean up (default: e)",
    )
    args = parser.parse_args()

    client = FirebaseClient()
    try:
        client.abort_game(args.room)
        print(f"Room '{args.room}': ゲームをリセットし、全席を退席しました")
    finally:
        client.close()


if __name__ == "__main__":
    main()
