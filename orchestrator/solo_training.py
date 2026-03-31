"""
Solo training orchestrator.

Runs PPO in single-player mode (no opponent).
P2 mirrors P1 actions so the environment remains two-sided internally,
but only the P1 policy is trained.

Usage:
    python -m orchestrator.solo_training
    python -m orchestrator.solo_training --arch cnn
"""
import sys
import time
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.env import PuyotanVectorEnv
from training.trainer import PPOTrainer
from training.model import ModelArch
from training.export import export_to_onnx
from training.config import DEFAULT_NUM_ENVS, DEFAULT_STEPS_PER_ITER

# ---------------------------------------------------------------------------
# Solo training configuration
# ---------------------------------------------------------------------------
NUM_ENVS       = DEFAULT_NUM_ENVS
STEPS_PER_ITER = DEFAULT_STEPS_PER_ITER
TOTAL_ITERS    = 500
LOG_INTERVAL   = 10
SAVE_INTERVAL  = 50

MODELS_DIR  = BASE_DIR / "models"
TIMESTAMP   = time.strftime("%Y%m%d_%H%M%S")


def solo_training_loop(
    config_name: str = "reward_solo.json",
    arch: ModelArch = ModelArch.MLP,
) -> None:
    print(f"=== PuyotanAI Solo Training  session={TIMESTAMP} ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.value.upper()}")

    # Define architecture-specific directories and paths
    arch_dir   = MODELS_DIR / arch.value
    latest_pt  = arch_dir / "puyotan_solo_latest.pt"
    session_pt = arch_dir / f"puyotan_solo_{TIMESTAMP}.pt"

    arch_dir.mkdir(parents=True, exist_ok=True)

    env = PuyotanVectorEnv(num_envs=NUM_ENVS)

    # Load reward weights - file is opened inside C++ via std::filesystem (UTF-8 safe)
    reward_config_path = BASE_DIR / "native" / "resources" / config_name
    env.reward_calc.load_from_json(str(reward_config_path))

    trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER, arch=arch)

    # Resume from latest checkpoint if available
    if latest_pt.exists():
        print(f"Resuming from checkpoint: {latest_pt}")
        trainer.load(str(latest_pt))

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    acc_loss = acc_sps = acc_avg_max = acc_reward = acc_score = 0.0
    acc_max_chain = 0.0

    for i in range(TOTAL_ITERS):
        iteration  = i + 1
        t0 = time.perf_counter()

        # Solo mode: no opponent policy
        metrics = trainer.train(p2_policy=None)

        elapsed = time.perf_counter() - t0
        sps     = (NUM_ENVS * STEPS_PER_ITER) / elapsed

        acc_loss        += metrics["loss"]
        acc_sps         += sps
        acc_avg_max     += metrics["avg_max_chain"]
        acc_reward      += metrics["avg_reward"]
        acc_score       += metrics["avg_score"]
        acc_max_chain    = max(acc_max_chain, metrics["max_chain"])

        if iteration % LOG_INTERVAL == 0 or iteration == 1:
            div = min(iteration, LOG_INTERVAL)
            print(
                f"[Iter {iteration:4d}/{TOTAL_ITERS}]"
                f"  AvgRew={acc_reward/div:6.3f}"
                f"  AvgScore={acc_score/div:6.1f}"
                f"  AvgMax={acc_avg_max/div:4.2f}"
                f"  SPS={acc_sps/div:.0f}"
            )
            acc_loss = acc_sps = acc_max_chain = acc_avg_max = 0.0
            acc_reward = acc_score = 0.0

        if iteration % SAVE_INTERVAL == 0:
            trainer.save(str(session_pt))
            trainer.save(str(latest_pt))
            print(f"Saved: {session_pt.name}  (and {latest_pt.name})")

            model_raw    = trainer.model._orig_mod if hasattr(trainer.model, "_orig_mod") else trainer.model
            onnx_session = session_pt.with_suffix(".onnx")
            onnx_latest  = latest_pt.with_suffix(".onnx")
            export_to_onnx(model_raw, str(onnx_session))
            shutil.copy2(onnx_session, onnx_latest)
            print(f"Exported ONNX: {onnx_session.name}  (and {onnx_latest.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuyotanAI solo training")
    parser.add_argument("--arch",   default="mlp", choices=["mlp", "cnn"],
                        help="Backbone architecture (default: mlp)")
    parser.add_argument("--reward", default="reward_solo.json",
                        help="Reward config filename under native/resources/")
    args = parser.parse_args()

    solo_training_loop(
        config_name=args.reward,
        arch=ModelArch(args.arch),
    )
