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
STEPS_PER_ITER = 128
TOTAL_ITERS = 1000
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PT = MODELS_DIR / "puyotan_latest.pt"
CHECKPOINT_ONNX = MODELS_DIR / "puyotan_latest.onnx"

class RandomPolicy:
    def __init__(self):
        self.is_loaded = lambda: True
        
    def infer(self, obs, num_envs):
        # 22 actions
        return np.random.randint(0, 22, size=num_envs).astype(np.int32)

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
        
        # Policies
        random_policy = RandomPolicy()
        past_policy = None
        
        # 2. メインループ
        for i in range(TOTAL_ITERS):
            start_time = time.time()
            iteration = i + 1
            print(f"\n--- Iteration {iteration}/{TOTAL_ITERS} ---")
            
            # Curriculum: Stage 1 = PASS, Stage 2 = Random, Stage 3 = Self-Play
            p2_policy = None
            stage_name = "Stage 1 (PASS)"
            
            if iteration > 50 and iteration <= 150:
                p2_policy = random_policy
                stage_name = "Stage 2 (Random)"
            elif iteration > 150:
                # Update snapshot every 50 iterations
                if iteration % 50 == 1 or past_policy is None:
                    print(f"Exporting snapshot for Self-Play...")
                    trainer.save(str(CHECKPOINT_PT))
                    export_to_onnx(str(CHECKPOINT_PT), str(CHECKPOINT_ONNX))
                    
                    import shutil
                    import tempfile
                    try:
                        # Workaround for ONNX Runtime Japanese path issues
                        temp_onnx = Path(tempfile.gettempdir()) / "puyotan_snapshot.onnx"
                        shutil.copy2(CHECKPOINT_ONNX, temp_onnx)
                        
                        data_path = Path(CHECKPOINT_ONNX).with_suffix(".onnx.data")
                        if data_path.exists():
                            shutil.copy2(data_path, temp_onnx.with_suffix(".onnx.data"))
                            
                        past_policy = p.OnnxPolicy(str(temp_onnx), use_cpu=True)
                        if not past_policy.is_loaded():
                            print("Failed to load ONNX. Falling back to Random.")
                            past_policy = random_policy
                    except Exception as e:
                        print(f"Load Error: {e}")
                        past_policy = random_policy
                        
                p2_policy = past_policy
                stage_name = "Stage 3 (Self-Play)"
                
            print(f"Opponent: {stage_name}")
            
            # 学習実行
            metrics = trainer.train(num_steps=STEPS_PER_ITER, p2_policy=p2_policy)
            
            elapsed = time.time() - start_time
            fps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
            print(f"Iteration {iteration} Finished. Loss={metrics['loss']:.4f}, MaxChain={metrics['max_chain']}, FPS={fps:.1f}")
            
            # 定期的なモデル保存
            if iteration % 10 == 0:
                trainer.save(str(CHECKPOINT_PT))
            
    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    selfplay_loop()
