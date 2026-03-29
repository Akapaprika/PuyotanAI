import os
import time
import torch
import numpy as np
from pathlib import Path
import sys
import shutil
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
STEPS_PER_ITER = 128
TOTAL_ITERS = 1000
LOG_INTERVAL = 10                     # ログ出力間隔（イテレーション数）
SAVE_INTERVAL = 10                    # モデル保存間隔
MODELS_DIR = BASE_DIR / "models"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
# 最新版 (学習継続用)
LATEST_PT = MODELS_DIR / "puyotan_latest.pt"
LATEST_ONNX = MODELS_DIR / "puyotan_latest.onnx"
# セッション固有名 (記録用)
SESSION_PT = MODELS_DIR / f"puyotan_selfplay_{TIMESTAMP}.pt"
SESSION_ONNX = MODELS_DIR / f"puyotan_selfplay_{TIMESTAMP}.onnx"

class RandomPolicy:
    def __init__(self):
        self.is_loaded = lambda: True
        
    def infer(self, obs, num_envs):
        return np.random.randint(0, 22, size=num_envs).astype(np.int32)

def selfplay_loop(config_name="reward_match.json"):
    print("=== PuyotanAI Self-Play & Training Loop Starting ===")
    print(f"Using reward config: {config_name}")
    MODELS_DIR.mkdir(exist_ok=True)
    
    try:
        env = PuyotanVectorEnv(num_envs=NUM_ENVS)
        
        # 中央管理された報酬設定をロード (C++ 側でファイル読み込み完結)
        reward_config_path = BASE_DIR / "native" / "resources" / config_name
        env.reward_calc.load_from_json(str(reward_config_path))
        
        trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER)
        
        if LATEST_PT.exists():
            print(f"Loading existing checkpoint: {LATEST_PT}")
            trainer.load(str(LATEST_PT))
        elif (MODELS_DIR / "puyotan_latest.pt").exists():
            # 移行用
            print(f"Loading legacy checkpoint: puyotan_latest.pt")
            trainer.load(str(MODELS_DIR / "puyotan_latest.pt"))
        
        print(f"Training Config: envs={NUM_ENVS}, steps={STEPS_PER_ITER}, "
              f"log_interval={LOG_INTERVAL}")
        
        random_policy = RandomPolicy()
        past_policy = None
        
        # 集計用（LOG_INTERVAL 毎にまとめて出力）
        acc_loss = 0.0
        acc_sps = 0.0
        acc_mean_chain = 0.0
        acc_avg_max_chain = 0.0
        acc_max_chain = 0.0
        acc_reward = 0.0
        acc_score = 0.0
        
        for i in range(TOTAL_ITERS):
            start_time = time.perf_counter()  # time.time() より高精度
            iteration = i + 1
            
            # Self-Play: Always play against a previous version of itself
            if iteration % 50 == 1 or past_policy is None:
                trainer.save(str(SESSION_PT))
                trainer.save(str(LATEST_PT))
                model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
                export_to_onnx(model_for_export, str(SESSION_ONNX))
                shutil.copy2(SESSION_ONNX, LATEST_ONNX)
                
                import tempfile
                try:
                    temp_onnx = Path(tempfile.gettempdir()) / "puyotan_snapshot.onnx"
                    shutil.copy2(LATEST_ONNX, temp_onnx)
                    
                    data_path = Path(LATEST_ONNX).with_suffix(".onnx.data")
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
            sps = (NUM_ENVS * STEPS_PER_ITER) / elapsed
            
            # 集計
            acc_loss += metrics['loss']
            acc_sps += sps
            acc_mean_chain += metrics['mean_chain']
            acc_avg_max_chain += metrics['avg_max_chain']
            acc_reward += metrics['avg_reward']
            acc_score += metrics['avg_score']
            mc = metrics['max_chain']
            if mc > acc_max_chain:
                acc_max_chain = mc
            
            # LOG_INTERVAL ごとにまとめて出力
            if iteration % LOG_INTERVAL == 0 or iteration == 1:
                avg_loss = acc_loss / min(iteration, LOG_INTERVAL)
                avg_sps = acc_sps / min(iteration, LOG_INTERVAL)
                avg_chain = acc_mean_chain / min(iteration, LOG_INTERVAL)
                avg_max = acc_avg_max_chain / min(iteration, LOG_INTERVAL)
                avg_reward = acc_reward / min(iteration, LOG_INTERVAL)
                avg_score = acc_score / min(iteration, LOG_INTERVAL)
                
                print(f"[Iter {iteration:4d}/{TOTAL_ITERS}] "
                      f"AvgRew={avg_reward:6.3f} | "
                      f"AvgScore={avg_score:6.1f} | "
                      f"AvgChain={avg_chain:4.2f} | "
                      f"AvgMax={avg_max:4.2f} | "
                      f"Max={acc_max_chain:2d} | "
                      f"SPS={avg_sps:.0f}")
                
                acc_loss = 0.0
                acc_sps = 0.0
                acc_max_chain = 0
                acc_mean_chain = 0.0
                acc_avg_max_chain = 0.0
                acc_reward = 0.0
                acc_score = 0.0
            
            # 定期的なモデル保存
            if iteration % SAVE_INTERVAL == 0:
                trainer.save(str(SESSION_PT))
                trainer.save(str(LATEST_PT))
                # Export ONNX as well for immediate GUI usage
                model_for_export = trainer.model._orig_mod if hasattr(trainer.model, '_orig_mod') else trainer.model
                export_to_onnx(model_for_export, str(SESSION_ONNX))
                shutil.copy2(SESSION_ONNX, LATEST_ONNX)
            
    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    selfplay_loop()
