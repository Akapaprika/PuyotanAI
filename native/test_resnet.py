import sys
import os
from pathlib import Path

BASE_DIR = Path(r"g:\マイドライブ\資料\プログラミング\プロジェクト\PuyotanAI")
DIST_DIR = BASE_DIR / "native" / "dist"
if os.name == "nt" and DIST_DIR.exists():
    os.add_dll_directory(str(DIST_DIR))
sys.path.insert(0, str(DIST_DIR))

import puyotan_native
print("Loaded puyotan_native compiled binary.")

cfg = puyotan_native.PPOConfig()
cfg.minibatch = 16

print("Instantiating ResNet trainer...")
trainer = puyotan_native.CppPPOTrainer(
    num_envs=16, 
    num_steps=10, 
    arch="resnet", 
    hidden_dim=128, 
    base_seed=1, 
    cfg=cfg
)

print("Running one training step...")
metrics = trainer.trainStep(False)
print(f"ResNet Loss: {metrics.loss:.4f}")
print(f"Avg Reward : {metrics.avg_reward:.4f}")
print("Test completed successfully.")
