"""
Self-play training orchestrator.

The current model (P1) trains against a frozen snapshot of itself (P2).
Every SNAPSHOT_INTERVAL iterations the snapshot is refreshed so P2 gradually
catches up, preventing P1 from over-specializing against a static opponent.

Usage:
    python -m orchestrator.selfplay
    python -m orchestrator.selfplay --arch cnn
"""
import sys
import time
import shutil
import tempfile
import argparse
import traceback
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.env import PuyotanVectorEnv
import puyotan_native as p
from training.trainer import PPOTrainer
from training.model import ModelArch
from training.export import export_to_onnx
from training.config import DEFAULT_NUM_ENVS, DEFAULT_STEPS_PER_ITER

# ---------------------------------------------------------------------------
# Self-play training configuration
# ---------------------------------------------------------------------------
NUM_ENVS          = DEFAULT_NUM_ENVS
STEPS_PER_ITER    = DEFAULT_STEPS_PER_ITER
TOTAL_ITERS       = 1000
LOG_INTERVAL      = 10
SAVE_INTERVAL     = 10
SNAPSHOT_INTERVAL = 50   # How often the frozen opponent policy is refreshed

MODELS_DIR   = BASE_DIR / "models"
TIMESTAMP    = time.strftime("%Y%m%d_%H%M%S")


class _RandomPolicy:
    """Fallback opponent that selects uniformly random actions."""
    def is_loaded(self) -> bool:
        return True

    def infer(self, obs, num_envs: int) -> np.ndarray:
        return np.random.randint(0, 22, size=num_envs).astype(np.int32)


def _export_snapshot(trainer: PPOTrainer, pt_path: Path, onnx_path: Path, latest_pt: Path, latest_onnx: Path) -> None:
    trainer.save(str(pt_path))
    model_raw = trainer.model._orig_mod if hasattr(trainer.model, "_orig_mod") else trainer.model
    export_to_onnx(model_raw, str(onnx_path))
    shutil.copy2(onnx_path, latest_onnx)
    shutil.copy2(pt_path,   latest_pt)


def _load_opponent_policy(onnx_path: Path, fallback) -> object:
    """Load an OnnxPolicy from a temp copy; fall back to random on failure."""
    try:
        tmp = Path(tempfile.gettempdir()) / "puyotan_snapshot.onnx"
        shutil.copy2(onnx_path, tmp)
        data = onnx_path.with_suffix(".onnx.data")
        if data.exists():
            shutil.copy2(data, tmp.with_suffix(".onnx.data"))
        policy = p.OnnxPolicy(str(tmp), use_cpu=True)
        return policy if policy.is_loaded() else fallback
    except Exception as exc:
        print(f"[WARN] Failed to load opponent ONNX: {exc}")
        return fallback


def selfplay_loop(
    config_name: str = "reward_match.json",
    arch: ModelArch = ModelArch.MLP,
) -> None:
    print("=== PuyotanAI Self-Play Training ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.value.upper()}")

    # Define architecture-specific directories and paths
    arch_dir     = MODELS_DIR / arch.value
    latest_pt    = arch_dir / "puyotan_latest.pt"
    latest_onnx  = arch_dir / "puyotan_latest.onnx"
    session_pt   = arch_dir / f"puyotan_selfplay_{TIMESTAMP}.pt"
    session_onnx = arch_dir / f"puyotan_selfplay_{TIMESTAMP}.onnx"

    arch_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = PuyotanVectorEnv(num_envs=NUM_ENVS)

        # Load reward weights - file is opened inside C++ via std::filesystem (UTF-8 safe)
        reward_config_path = BASE_DIR / "native" / "resources" / config_name
        env.reward_calc.load_from_json(str(reward_config_path))

        trainer = PPOTrainer(env, num_rollout_steps=STEPS_PER_ITER, arch=arch)

        if latest_pt.exists():
            print(f"Resuming from checkpoint: {latest_pt}")
            trainer.load(str(latest_pt))

        print(f"Config: envs={NUM_ENVS}  steps={STEPS_PER_ITER}  log_every={LOG_INTERVAL}")

        random_policy = _RandomPolicy()
        past_policy   = None

        # Accumulator reset helper
        def _reset_accumulators():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        acc_loss, acc_sps, acc_mean_chain, acc_avg_max, acc_reward, acc_score = _reset_accumulators()
        acc_max_chain = 0

        for i in range(TOTAL_ITERS):
            iteration = i + 1

            # Refresh frozen opponent snapshot periodically
            if iteration % SNAPSHOT_INTERVAL == 1 or past_policy is None:
                _export_snapshot(trainer, session_pt, session_onnx, latest_pt, latest_onnx)
                past_policy = _load_opponent_policy(latest_onnx, random_policy)

            t0      = time.perf_counter()
            metrics = trainer.train(p2_policy=past_policy)
            elapsed = time.perf_counter() - t0
            sps     = (NUM_ENVS * STEPS_PER_ITER) / elapsed

            acc_loss      += metrics["loss"]
            acc_sps       += sps
            acc_avg_max   += metrics["avg_max_chain"]
            acc_reward    += metrics["avg_reward"]
            acc_score     += metrics["avg_score"]
            acc_max_chain  = max(acc_max_chain, metrics["max_chain"])

            if iteration % LOG_INTERVAL == 0 or iteration == 1:
                div = min(iteration, LOG_INTERVAL)
                print(
                    f"[Iter {iteration:4d}/{TOTAL_ITERS}]"
                    f"  AvgRew={acc_reward/div:6.3f}"
                    f"  AvgScore={acc_score/div:6.1f}"
                    f"  AvgMax={acc_avg_max/div:4.2f}"
                    f"  Max={acc_max_chain:2d}"
                    f"  SPS={acc_sps/div:.0f}"
                )
                acc_loss, acc_sps, acc_mean_chain, acc_avg_max, acc_reward, acc_score = _reset_accumulators()
                acc_max_chain = 0

            if iteration % SAVE_INTERVAL == 0:
                _export_snapshot(trainer, session_pt, session_onnx, latest_pt, latest_onnx)
                print(f"Saved: {session_pt.name}  ONNX: {session_onnx.name}")

    except Exception:
        print("CRITICAL ERROR in selfplay_loop:")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuyotanAI self-play training")
    parser.add_argument("--arch",   default="mlp", choices=["mlp", "cnn"],
                        help="Backbone architecture (default: mlp)")
    parser.add_argument("--reward", default="reward_match.json",
                        help="Reward config filename under native/resources/")
    args = parser.parse_args()

    selfplay_loop(
        config_name=args.reward,
        arch=ModelArch(args.arch),
    )
