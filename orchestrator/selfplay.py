"""
Self-play training orchestrator.

The current model (P1) trains against a frozen snapshot of itself (P2)
using high-performance C++ LibTorch integration natively.
Every SNAPSHOT_INTERVAL iterations the snapshot is refreshed.

Usage:
    python orchestrator/selfplay.py
    python orchestrator/selfplay.py --arch cnn
"""
import sys
import os
import time
import shutil
import argparse
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DIST_DIR = BASE_DIR / "native" / "dist"
if os.name == "nt" and DIST_DIR.exists():
    os.add_dll_directory(str(DIST_DIR))
    os.environ["PATH"] = str(DIST_DIR) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(DIST_DIR))

try:
    import puyotan_native
except ImportError as e:
    print(f"Failed to import puyotan_native: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Self-play training configuration
# ---------------------------------------------------------------------------
NUM_ENVS          = 256
STEPS_PER_ITER    = 128
TOTAL_ITERS       = 1000
LOG_INTERVAL      = 10
SAVE_INTERVAL     = 50
SNAPSHOT_INTERVAL = 50   # How often the frozen opponent policy is refreshed
HIDDEN_DIM        = 128

MODELS_DIR = BASE_DIR / "models"
TIMESTAMP  = time.strftime("%Y%m%d_%H%M%S")


def selfplay_loop(
    config_name: str = "reward_match.json",
    arch: str = "mlp",
) -> None:
    print(f"=== PuyotanAI Self-Play Training (C++ LibTorch) session={TIMESTAMP} ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.upper()}")

    arch_dir   = MODELS_DIR / arch
    latest_pt  = arch_dir / "puyotan_latest.pt"
    session_pt = arch_dir / f"puyotan_selfplay_{TIMESTAMP}.pt"

    arch_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = puyotan_native.PPOConfig()
        cfg.lr = 3e-4
        cfg.num_epochs = 4
        cfg.minibatch = 8192

        trainer = puyotan_native.CppPPOTrainer(
            num_envs=NUM_ENVS,
            num_steps=STEPS_PER_ITER,
            arch=arch,
            hidden_dim=HIDDEN_DIM,
            base_seed=1,
            cfg=cfg,
        )

        reward_config_path = BASE_DIR / "native" / "resources" / config_name
        trainer.env.reward_calc.load_from_json(str(reward_config_path))

        # UTF-8 paths are now handled natively in C++ — no temporary file needed.
        if latest_pt.exists():
            print(f"Resuming from checkpoint: {latest_pt}")
            trainer.load(str(latest_pt))

        print(f"Config: envs={NUM_ENVS}  steps={STEPS_PER_ITER}  log_every={LOG_INTERVAL}")

        def _reset_accumulators():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        acc_loss, acc_sps, acc_mean_chain, acc_avg_max, acc_reward, acc_score = _reset_accumulators()
        acc_max_chain = 0

        for i in range(TOTAL_ITERS):
            iteration = i + 1

            # Refresh frozen opponent snapshot periodically.
            if iteration % SNAPSHOT_INTERVAL == 1:
                trainer.save(str(latest_pt))
                trainer.loadP2(str(latest_pt))

            t0 = time.perf_counter()

            # If no snapshot yet (iteration 1 before save), fall back to random P2.
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
                shutil.copy2(str(session_pt), str(latest_pt))
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
