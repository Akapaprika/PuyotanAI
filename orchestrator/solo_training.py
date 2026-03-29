import os
import time
import torch
import numpy as np
from pathlib import Path
import sys
import shutil

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.env import PuyotanVectorEnv
from training.trainer import PPOTrainer
from training.export import export_to_onnx

# 設定
NUM_ENVS = 256
STEPS_PER_ITER = 128
TOTAL_ITERS = 500  # ソロ用にとりあえず短めに設定
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
MODELS_DIR = BASE_DIR / "models"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
# 固定名 (レジューム用)
LATEST_PT = MODELS_DIR / "puyotan_solo_latest.pt"
# セッション固有名 (記録用)
SESSION_PT = MODELS_DIR / f"puyotan_solo_{TIMESTAMP}.pt"

def solo_training_loop():
    print(f"=== PuyotanAI Solo Training Starting (Session: {TIMESTAMP}) ===")
    MODELS_DIR.mkdir(exist_ok=True)
    
    env = PuyotanVectorEnv(num_envs=NUM_ENVS)
    
    # 中央管理された報酬設定をロード (Python側で読み込んで文字列として渡すことで日本語パス問題を回避)
    reward_config_path = BASE_DIR / "native" / "resources" / "reward_default.json"
    if reward_config_path.exists():
        with open(reward_config_path, "r", encoding="utf-8") as f:
            env.reward_calc.load_from_json_string(f.read())
    else:
        print(f"[WARNING] Reward config not found: {reward_config_path}")
    
    trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER)
    
    if LATEST_PT.exists():
        print(f"Loading existing solo checkpoint: {LATEST_PT}")
        trainer.load(str(LATEST_PT))
    elif (MODELS_DIR / "puyotan_solo.pt").exists():
        # 移行用：古い名前があればロード
        print(f"Loading legacy checkpoint: puyotan_solo.pt")
        trainer.load(str(MODELS_DIR / "puyotan_solo.pt"))
    
    # 集計用
    acc_loss = 0.0
    acc_sps = 0.0
    acc_avg_max_chain = 0.0
    acc_max_chain = 0.0
    acc_reward = 0.0
    acc_score = 0.0
    
    for i in range(TOTAL_ITERS):
        start_time = time.perf_counter()
        iteration = i + 1
        
        # ソロモードなので p2_policy は常に None
        metrics = trainer.train(p2_policy=None)
        
        elapsed = time.perf_counter() - start_time
        sps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
        
        # 集計
        acc_loss += metrics['loss']
        acc_sps += sps
        acc_avg_max_chain += metrics['avg_max_chain']
        acc_reward += metrics['avg_reward']
        acc_score += metrics['avg_score']
        if metrics['max_chain'] > acc_max_chain:
            acc_max_chain = metrics['max_chain']
        
        if iteration % LOG_INTERVAL == 0 or iteration == 1:
            div = min(iteration, LOG_INTERVAL)
            avg_loss   = acc_loss / div
            avg_sps    = acc_sps / div
            avg_max    = acc_avg_max_chain / div
            avg_reward = acc_reward / div
            avg_score  = acc_score / div
            
            print(f"[Iter {iteration:4d}/{TOTAL_ITERS}] "
                  f"AvgRew={avg_reward:6.3f} | "
                  f"AvgScore={avg_score:6.1f} | "
                  f"AvgMax={avg_max:4.2f} | "
                  f"SPS={avg_sps:.0f}")
            
            acc_loss, acc_sps, acc_max_chain, acc_avg_max_chain = 0, 0, 0, 0
            acc_reward, acc_score = 0, 0
        
        if iteration % SAVE_INTERVAL == 0:
            trainer.save(str(SESSION_PT))
            trainer.save(str(LATEST_PT))
            print(f"Saved checkpoint: {SESSION_PT} (and {LATEST_PT.name})")
            
            # Export ONNX as well for immediate GUI usage
            model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
            
            onnx_session = SESSION_PT.with_suffix(".onnx")
            onnx_latest = LATEST_PT.with_suffix(".onnx")
            
            export_to_onnx(model_for_export, str(onnx_session))
            
            # Copy to latest for GUI reference
            shutil.copy2(onnx_session, onnx_latest)
            print(f"Exported ONNX: {onnx_session} (and {onnx_latest.name})")

if __name__ == "__main__":
    solo_training_loop()
