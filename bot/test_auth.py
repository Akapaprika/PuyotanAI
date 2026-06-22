"""
Bot 停止時の後片付けスクリプト。
着席中のBotを退席させ、進行中のゲームをリセットする。
"""
from bot.firebase_client import FirebaseClient

client = FirebaseClient()

# room e をクリーンアップ
client.abort_game("e")
print("room e: ゲームをリセットし、全席を退席しました")
