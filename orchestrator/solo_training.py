import os
import time
import torch
import numpy as np
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.env import PuyotanVectorEnv
from training.trainer import PPOTrainer
from training.export import export_to_onnx

# 設定
NUM_ENVS = 256
STEPS_PER_ITER = 192
TOTAL_ITERS = 500  # ソロ用にとりあえず短めに設定
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PT = MODELS_DIR / "puyotan_solo.pt"

def solo_training_loop():
    print("=== PuyotanAI Solo Training Starting (Score Attack Mode) ===")
    MODELS_DIR.mkdir(exist_ok=True)
    
    env = PuyotanVectorEnv(num_envs=NUM_ENVS)
    
    # 中央管理された報酬設定をロード (C++ ネイティブ)
    reward_config_path = BASE_DIR / "native" / "resources" / "reward_default.json"
    env.reward_calc.load_from_json(str(reward_config_path))
    
    trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER)
    
    if CHECKPOINT_PT.exists():
        print(f"Loading existing solo checkpoint: {CHECKPOINT_PT}")
        trainer.load(str(CHECKPOINT_PT))
    
    # 集計用
    acc_loss = 0.0
    acc_fps = 0.0
    acc_mean_chain = 0.0
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
        fps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
        
        # 集計
        acc_loss += metrics['loss']
        acc_fps += fps
        acc_mean_chain += metrics['mean_chain']
        acc_avg_max_chain += metrics['avg_max_chain']
        acc_reward += metrics['avg_reward']
        acc_score += metrics['avg_score']
        if metrics['max_chain'] > acc_max_chain:
            acc_max_chain = metrics['max_chain']
        
        if iteration % LOG_INTERVAL == 0 or iteration == 1:
            div = min(iteration, LOG_INTERVAL)
            avg_loss   = acc_loss / div
            avg_fps    = acc_fps / div
            avg_chain  = acc_mean_chain / div
            avg_max    = acc_avg_max_chain / div
            avg_reward = acc_reward / div
            avg_score  = acc_score / div
            
            print(f"[Iter {iteration:4d}/{TOTAL_ITERS}] "
                  f"AvgRew={avg_reward:6.3f} | "
                  f"AvgScore={avg_score:6.1f} | "
                  f"AvgChain={avg_chain:4.2f} | "
                  f"AvgMax={avg_max:4.2f} | "
                  f"FPS={avg_fps:.0f}")
            
            acc_loss, acc_fps, acc_max_chain, acc_mean_chain, acc_avg_max_chain = 0, 0, 0, 0, 0
            acc_reward, acc_score = 0, 0
        
        if iteration % SAVE_INTERVAL == 0:
            trainer.save(str(CHECKPOINT_PT))
            print(f"Saved checkpoint: {CHECKPOINT_PT}")
            # Export ONNX as well for immediate GUI usage
            model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
            export_to_onnx(model_for_export, str(CHECKPOINT_PT.with_suffix(".onnx")))

if __name__ == "__main__":
    solo_training_loop()
