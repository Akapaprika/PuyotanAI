"""
Solo training orchestrator.

Runs PPO in single-player mode (no opponent) using high-performance C++ LibTorch integration.
P2 mirrors P1 actions so the environment remains two-sided internally,
but only the P1 policy is trained.

Usage:
    python -m orchestrator.solo_training
    python -m orchestrator.solo_training --arch cnn
"""
import sys
import os
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DIST_DIR = BASE_DIR / "native" / "dist"
if os.name == "nt" and DIST_DIR.exists():
    os.add_dll_directory(str(DIST_DIR))
sys.path.insert(0, str(DIST_DIR))

"""
Solo training orchestrator.

Runs PPO in single-player mode (no opponent) using high-performance C++ LibTorch integration.
P2 mirrors P1 actions so the environment remains two-sided internally,
but only the P1 policy is trained.

Usage:
    python -m orchestrator.solo_training
    python -m orchestrator.solo_training --arch cnn
"""
import sys
import os
import time
import argparse
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
# Solo training configuration
# ---------------------------------------------------------------------------
NUM_ENVS       = DEFAULT_NUM_ENVS
STEPS_PER_ITER = DEFAULT_STEPS_PER_ITER
TOTAL_ITERS    = 500
LOG_INTERVAL   = 10
SAVE_INTERVAL  = 50
HIDDEN_DIM     = 128

MODELS_DIR  = BASE_DIR / "models"
TIMESTAMP   = time.strftime("%Y%m%d_%H%M%S")


def solo_training_loop(
    config_name: str = "reward_solo.json",
    arch: str = "mlp",
) -> None:
    print(f"=== PuyotanAI Solo Training (C++ LibTorch) session={TIMESTAMP} ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.upper()}")
    print(f"  envs          : {NUM_ENVS}")
    print(f"  steps/iter    : {STEPS_PER_ITER}")

    # Define architecture-specific directories and paths
    arch_dir   = MODELS_DIR / arch
    latest_pt  = arch_dir / "puyotan_solo_latest.pt"
    session_pt = arch_dir / f"puyotan_solo_{TIMESTAMP}.pt"

    arch_dir.mkdir(parents=True, exist_ok=True)

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
    SAFE_MODEL_PATH = "cpp_tmp_model_solo.pt"

    if latest_pt.exists():
        print(f"Resuming from checkpoint: {latest_pt}")
        shutil.copy2(str(latest_pt), SAFE_MODEL_PATH)
        trainer.load(SAFE_MODEL_PATH)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    acc_loss = acc_sps = acc_avg_max = acc_reward = acc_score = 0.0
    acc_max_chain = 0.0

    for i in range(TOTAL_ITERS):
        iteration  = i + 1
        t0 = time.perf_counter()

        # Solo mode: p2 does not play random
        metrics = trainer.trainStep(p2_random=False)

        elapsed = time.perf_counter() - t0
        sps     = (NUM_ENVS * STEPS_PER_ITER) / elapsed

        acc_loss        += metrics.loss
        acc_sps         += sps
        acc_avg_max     += metrics.avg_max_chain
        acc_reward      += metrics.avg_reward
        acc_score       += metrics.avg_score
        acc_max_chain    = max(acc_max_chain, metrics.max_chain)

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
            acc_loss = acc_sps = acc_max_chain = acc_avg_max = acc_reward = acc_score = 0.0

        if iteration % SAVE_INTERVAL == 0 or iteration == TOTAL_ITERS:
            trainer.save(SAFE_MODEL_PATH)
            shutil.copy2(SAFE_MODEL_PATH, str(session_pt))
            shutil.copy2(SAFE_MODEL_PATH, str(latest_pt))
            print(f"Saved checkpoint -> {session_pt.name} and {latest_pt.name}")

    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuyotanAI Solo Training (C++)")
    parser.add_argument("--config", type=str, default="reward_solo.json", help="Reward weights JSON name")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "cnn", "attention"], help="Model backbone architecture")
    args = parser.parse_args()

    solo_training_loop(config_name=args.config, arch=args.arch)
