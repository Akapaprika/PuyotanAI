import os
import sys
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.env import PuyotanVectorEnv
from training.model import PuyotanPolicy

def print_board(field_array):
    """
    field_array: [2, 5, 6, 14] or similar
    P1の盤面を表示する (player_id=0)
    """
    # 実際の色名マッピング
    colors = {0: ".", 1: "R", 2: "G", 3: "B", 4: "Y", 5: "O"}
    
    # [color, row, col] に変換
    # 観測データは [player, color, row, col]
    # color 0-4 が通常色, 5 がお邪魔
    
    # 盤面を再構築 (14行 6列)
    board = [["." for _ in range(6)] for _ in range(14)]
    
    # Row 0-12: Actual field
    for c in range(5): # Normal colors
        for col in range(6):
            for r in range(13): # Only up to 12
                if field_array[0, 0, c, col, r] > 0:
                    board[r][col] = colors[c+1]
    
    # お邪魔 (Row 0-12)
    for col in range(6):
        for r in range(13):
            if field_array[0, 0, 4, col, r] > 0:
                 board[r][col] = "X"

    # Row 13: Metadata (Next pieces & Ojama)
    next_pieces = []
    # Col 0, 1: Next1, Col 2, 3: Next2, Col 4, 5: Next3
    for col in range(6):
        found_color = "."
        for c in range(5):
            if field_array[0, 0, c, col, 13] > 0:
                found_color = colors[c+1]
                break
        next_pieces.append(found_color)

    active_ojama = int(field_array[0, 0, 0, 0, 13])
    non_active_ojama = int(field_array[0, 0, 0, 1, 13])

    print("  0 1 2 3 4 5")
    for r in reversed(range(13)): # Only show 0-12
        row_str = f"{r:2d} " + " ".join(board[r])
        print(row_str)
    print("  -----------")
    print(f"  Next: ({next_pieces[0]}-{next_pieces[1]}) ({next_pieces[2]}-{next_pieces[3]}) ({next_pieces[4]}-{next_pieces[5]})")
    print(f"  Ojama: Active={active_ojama}, Non-Active={non_active_ojama}")

def watch_solo(model_path="models/puyotan_latest.pt"):
    print(f"=== Watching Solo Play using {model_path} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PuyotanVectorEnv(num_envs=1)
    model = PuyotanPolicy(hidden_dim=128).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        print(f"Warning: {model_path} not found. Using random initialized model.")
    
    model.eval()
    obs, _ = env.reset()
    
    total_reward = 0
    step = 0
    
    try:
        while step < 64: # 最大64手分表示
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_t)
            
            act = action.cpu().numpy().astype(np.int32)
            obs, rewards, dones, _, info = env.step(act, act)  # Mirror P1 for solo mode
            
            reward = rewards[0]
            total_reward += reward
            chains = info["chains"][0]
            
            print(f"\n[Step {step}] Action: {act[0]} | Reward: {reward:.4f} | Total: {total_reward:.4f}")
            if chains > 0:
                print(f"*** CHAIN! {chains} chains ***")
            
            print_board(obs)
            
            if dones[0]:
                print("Game Over (Topped out?)")
                break
                
            step += 1
            # 1秒待機すると見やすい（オプション）
            # time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    path = "models/puyotan_solo.pt"
    if not os.path.exists(path):
        path = "models/puyotan_latest.pt"
    watch_solo(path)
