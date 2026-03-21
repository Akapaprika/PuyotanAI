import os
import time
import torch
import numpy as np
from pathlib import Path
import sys
import traceback

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from puyotan_env import PuyotanVectorEnv
import puyotan_native as p
from training.trainer import PPOTrainer
from training.export import export_to_onnx

# 設定
NUM_ENVS = 128
STEPS_PER_ITER = 64
TOTAL_ITERS = 1000
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PT = MODELS_DIR / "puyotan_latest.pt"
CHECKPOINT_ONNX = MODELS_DIR / "puyotan_latest.onnx"

def selfplay_loop():
    print("=== PuyotanAI Self-Play & Training Loop Starting ===")
    MODELS_DIR.mkdir(exist_ok=True)
    
    try:
        # 1. 環境とトレーナーの初期化
        env = PuyotanVectorEnv(num_envs=NUM_ENVS)
        trainer = PPOTrainer(env)
        
        # 既存のチェックポイントがあればロード
        if CHECKPOINT_PT.exists():
            print(f"Loading existing checkpoint: {CHECKPOINT_PT}")
            trainer.load(str(CHECKPOINT_PT))
        
        print(f"Training Config: envs={NUM_ENVS}, steps={STEPS_PER_ITER}")
        
        # 2. メインループ
        for i in range(TOTAL_ITERS):
            start_time = time.time()
            iteration = i + 1
            print(f"\n--- Iteration {iteration}/{TOTAL_ITERS} ---")
            
            # 対戦相手の決定
            p2_policy = None
            # Stage 1: P2 = PASS (Learning the basics)
            
            # 学習実行
            metrics = trainer.train(num_steps=STEPS_PER_ITER, p2_policy=p2_policy)
            
            elapsed = time.time() - start_time
            fps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
            print(f"Iteration {iteration} Finished. Loss={metrics['loss']:.4f}, FPS={fps:.1f}")
            
            # 定期的なモデル保存 (とりあえず毎回)
            # trainer.save(str(CHECKPOINT_PT))
            
    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    selfplay_loop()
