"""
Self-play training orchestrator.

The current model (P1) trains against a frozen snapshot of itself (P2)
using high-performance C++ LibTorch integration natively.
Every SNAPSHOT_INTERVAL iterations the snapshot is refreshed.

Usage:
    python -m orchestrator.selfplay
    python -m orchestrator.selfplay --arch cnn
"""
import sys
import os
import time
import shutil
import tempfile
import argparse
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DIST_DIR = BASE_DIR / "native" / "dist"
if os.name == "nt" and DIST_DIR.exists():
    os.add_dll_directory(str(DIST_DIR))
sys.path.insert(0, str(DIST_DIR))

try:
    import puyotan_native
except ImportError as e:
    print(f"Failed to import puyotan_native: {e}")
    sys.exit(1)

from training.config import DEFAULT_NUM_ENVS, DEFAULT_STEPS_PER_ITER

# ---------------------------------------------------------------------------
# Self-play training configuration
# ---------------------------------------------------------------------------
NUM_ENVS          = DEFAULT_NUM_ENVS
STEPS_PER_ITER    = DEFAULT_STEPS_PER_ITER
TOTAL_ITERS       = 1000
LOG_INTERVAL      = 10
SAVE_INTERVAL     = 50
SNAPSHOT_INTERVAL = 50   # How often the frozen opponent policy is refreshed
HIDDEN_DIM        = 128

MODELS_DIR   = BASE_DIR / "models"
TIMESTAMP    = time.strftime("%Y%m%d_%H%M%S")

def selfplay_loop(
    config_name: str = "reward_match.json",
    arch: str = "mlp",
) -> None:
    print(f"=== PuyotanAI Self-Play Training (C++ LibTorch) session={TIMESTAMP} ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.upper()}")

    # Define architecture-specific directories and paths
    arch_dir     = MODELS_DIR / arch
    latest_pt    = arch_dir / "puyotan_latest.pt"
    session_pt   = arch_dir / f"puyotan_selfplay_{TIMESTAMP}.pt"

    arch_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize C++ PPO Config
        cfg = puyotan_native.PPOConfig()
        cfg.lr = 3e-4
        cfg.num_epochs = 4
        cfg.minibatch = 8192

        # Initialize Native C++ PPO Trainer
        trainer = puyotan_native.CppPPOTrainer(
            num_envs=NUM_ENVS,
            num_steps=STEPS_PER_ITER,
            arch=arch,
            hidden_dim=HIDDEN_DIM,
            base_seed=1,
            cfg=cfg
        )

        # Load reward weights
        reward_config_path = BASE_DIR / "native" / "resources" / config_name
        trainer.env.reward_calc.load_from_json(str(reward_config_path))
    
        # [Workaround for LibTorch UTF-8 Bug on Windows]
        # We use local ASCII filename for C++ load/save, then move it via Python
        SAFE_MODEL_PATH = "cpp_tmp_model.pt"

        if latest_pt.exists():
            print(f"Resuming from checkpoint: {latest_pt}")
            shutil.copy2(str(latest_pt), SAFE_MODEL_PATH)
            trainer.load(SAFE_MODEL_PATH)

        print(f"Config: envs={NUM_ENVS}  steps={STEPS_PER_ITER}  log_every={LOG_INTERVAL}")

        # Accumulator reset helper
        def _reset_accumulators():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        acc_loss, acc_sps, acc_mean_chain, acc_avg_max, acc_reward, acc_score = _reset_accumulators()
        acc_max_chain = 0

        for i in range(TOTAL_ITERS):
            iteration = i + 1

            # Refresh frozen opponent snapshot periodically natively!
            if iteration % SNAPSHOT_INTERVAL == 1:
                trainer.save(SAFE_MODEL_PATH)
                shutil.copy2(SAFE_MODEL_PATH, str(latest_pt))
                trainer.loadP2(SAFE_MODEL_PATH)

            t0      = time.perf_counter()
            
            # Since CppPPOTrainer uses opp_policy_ internals if instantiated via loadP2, p2_random=False makes it use opp
            # On first iteration if past_policy hasn't generated latest_pt, p2_random = True as a fallback
            metrics = trainer.trainStep(p2_random=not latest_pt.exists())
            
            elapsed = time.perf_counter() - t0
            sps     = (NUM_ENVS * STEPS_PER_ITER) / elapsed

            acc_loss      += metrics.loss
            acc_sps       += sps
            acc_avg_max   += metrics.avg_max_chain
            acc_reward    += metrics.avg_reward
            acc_score     += metrics.avg_score
            acc_max_chain  = max(acc_max_chain, metrics.max_chain)

            if iteration % LOG_INTERVAL == 0 or iteration == 1:
                div = min(iteration, LOG_INTERVAL)
                print(
                    f"[Iter {iteration:4d}/{TOTAL_ITERS}]"
                    f"  Loss={acc_loss/div:6.3f}"
                    f"  AvgRew={acc_reward/div:6.3f}"
                    f"  AvgScore={acc_score/div:6.1f}"
                    f"  AvgMax={acc_avg_max/div:4.2f}"
                    f"  Max={acc_max_chain:2d}"
                    f"  SPS={acc_sps/div:.0f}"
                )
                acc_loss, acc_sps, acc_mean_chain, acc_avg_max, acc_reward, acc_score = _reset_accumulators()
                acc_max_chain = 0

            if iteration % SAVE_INTERVAL == 0:
                trainer.save(str(session_pt))
                trainer.save(str(latest_pt))
                print(f"Saved: {session_pt.name}")

    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuyotanAI self-play training (C++)")
    parser.add_argument("--arch",   default="mlp", choices=["mlp", "cnn"],
                        help="Backbone architecture (default: mlp)")
    parser.add_argument("--reward", default="reward_match.json",
                        help="Reward config filename under native/resources/")
    args = parser.parse_args()

    selfplay_loop(
        config_name=args.reward,
        arch=args.arch,
    )
