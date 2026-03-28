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

from training.env import PuyotanVectorEnv
import puyotan_native as p
from training.trainer import PPOTrainer
from training.export import export_to_onnx

# 設定
NUM_ENVS = 256
STEPS_PER_ITER = 192
TOTAL_ITERS = 1000
LOG_INTERVAL = 10                     # ログ出力間隔（イテレーション数）
SAVE_INTERVAL = 10                    # モデル保存間隔
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PT = MODELS_DIR / "puyotan_latest.pt"
CHECKPOINT_ONNX = MODELS_DIR / "puyotan_latest.onnx"

class RandomPolicy:
    def __init__(self):
        self.is_loaded = lambda: True
        
    def infer(self, obs, num_envs):
        return np.random.randint(0, 22, size=num_envs).astype(np.int32)

def selfplay_loop():
    print("=== PuyotanAI Self-Play & Training Loop Starting ===")
    MODELS_DIR.mkdir(exist_ok=True)
    
    try:
        env = PuyotanVectorEnv(num_envs=NUM_ENVS)
        trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER)
        
        if CHECKPOINT_PT.exists():
            print(f"Loading existing checkpoint: {CHECKPOINT_PT}")
            trainer.load(str(CHECKPOINT_PT))
        
        print(f"Training Config: envs={NUM_ENVS}, steps={STEPS_PER_ITER}, "
              f"log_interval={LOG_INTERVAL}")
        
        random_policy = RandomPolicy()
        past_policy = None
        
        # 集計用（LOG_INTERVAL 毎にまとめて出力）
        acc_loss = 0.0
        acc_fps = 0.0
        acc_mean_chain = 0.0
        acc_avg_max_chain = 0.0
        acc_max_chain = 0.0
        
        for i in range(TOTAL_ITERS):
            start_time = time.perf_counter()  # time.time() より高精度
            iteration = i + 1
            
            # Curriculum: Stage 1 = PASS, Stage 2 = Random, Stage 3 = Self-Play
            p2_policy = None
            
            if iteration > 50 and iteration <= 150:
                p2_policy = random_policy
            elif iteration > 150:
                if iteration % 50 == 1 or past_policy is None:
                    trainer.save(str(CHECKPOINT_PT))
                    model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
                    export_to_onnx(model_for_export, str(CHECKPOINT_ONNX))
                    
                    import shutil
                    import tempfile
                    try:
                        temp_onnx = Path(tempfile.gettempdir()) / "puyotan_snapshot.onnx"
                        shutil.copy2(CHECKPOINT_ONNX, temp_onnx)
                        
                        data_path = Path(CHECKPOINT_ONNX).with_suffix(".onnx.data")
                        if data_path.exists():
                            shutil.copy2(data_path, temp_onnx.with_suffix(".onnx.data"))
                            
                        past_policy = p.OnnxPolicy(str(temp_onnx), use_cpu=True)
                        if not past_policy.is_loaded():
                            past_policy = random_policy
                    except Exception as e:
                        print(f"[WARN] ONNX Load Error: {e}")
                        past_policy = random_policy
                        
                p2_policy = past_policy
            
            # 学習実行
            metrics = trainer.train(p2_policy=p2_policy)
            
            elapsed = time.perf_counter() - start_time
            fps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
            
            # 集計
            acc_loss += metrics['loss']
            acc_fps += fps
            acc_mean_chain += metrics['mean_chain']
            acc_avg_max_chain += metrics['avg_max_chain']
            mc = metrics['max_chain']
            if mc > acc_max_chain:
                acc_max_chain = mc
            
            # LOG_INTERVAL ごとにまとめて出力（print のシステムコール削減）
            if iteration % LOG_INTERVAL == 0 or iteration == 1:
                stage = "PASS" if iteration <= 50 else ("Random" if iteration <= 150 else "Self-Play")
                avg_loss = acc_loss / min(iteration, LOG_INTERVAL)
                avg_fps = acc_fps / min(iteration, LOG_INTERVAL)
                avg_chain = acc_mean_chain / min(iteration, LOG_INTERVAL)
                avg_max = acc_avg_max_chain / min(iteration, LOG_INTERVAL)
                print(f"[Iter {iteration:4d}/{TOTAL_ITERS}] "
                      f"Stage={stage:9s} | "
                      f"AvgLoss={avg_loss:.4f} | "
                      f"AvgMax={avg_max:.2f} | "
                      f"AvgChain={avg_chain:.2f} | "
                      f"MaxChain={acc_max_chain} | "
                      f"AvgFPS={avg_fps:.0f}")
                acc_loss = 0.0
                acc_fps = 0.0
                acc_max_chain = 0
                acc_mean_chain = 0.0
                acc_avg_max_chain = 0.0
            
            # 定期的なモデル保存
            if iteration % SAVE_INTERVAL == 0:
                trainer.save(str(CHECKPOINT_PT))
                # Export ONNX as well for immediate GUI usage
                model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
                export_to_onnx(model_for_export, str(CHECKPOINT_ONNX))
            
    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    selfplay_loop()
