import os
# OS-level thread management to prevent CPU pipeline starvation and context-switching overhead on 2-core systems.
# Must be declared BEFORE any scientific/learning frameworks are imported.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
Consolidated training orchestrator for PuyotanAI.

Supports both solo training and selfplay training modes via command options.
Usage:
    python -m orchestrator.train --mode solo --arch resnet
    python -m orchestrator.train --mode selfplay --arch resnet
"""
import sys
import time
import random
import shutil
import argparse
import traceback
import subprocess
from pathlib import Path

try:
    from orchestrator import config
except ImportError:
    import config

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

MODELS_DIR = BASE_DIR / "models"
TIMESTAMP  = time.strftime("%Y%m%d_%H%M%S")


def run_training(
    mode: str = "solo",
    config_name: str = None,
    arch: str = "mlp",
) -> None:
    is_selfplay = (mode == "selfplay")
    cfg_inst = config.get_config(arch, is_selfplay=is_selfplay)

    if config_name is None:
        config_name = "reward_match.json" if is_selfplay else "reward_solo.json"

    print(f"=== PuyotanAI {mode.upper()} Training (C++ LibTorch) session={TIMESTAMP} ===")
    print(f"  reward config : {config_name}")
    print(f"  backbone      : {arch.upper()}")
    print(f"  envs          : {cfg_inst.NUM_ENVS}")
    print(f"  steps/iter    : {cfg_inst.STEPS_PER_ITER}")

    arch_dir   = MODELS_DIR / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Configure checkpoint paths
    if is_selfplay:
        latest_pt  = arch_dir / "puyotan_latest.pt"
        session_pt = arch_dir / f"puyotan_selfplay_{TIMESTAMP}.pt"
    else:
        latest_pt  = arch_dir / "puyotan_solo_latest.pt"
        session_pt = arch_dir / f"puyotan_solo_{TIMESTAMP}.pt"

    try:
        cfg = puyotan_native.PPOConfig()
        cfg.lr = cfg_inst.LEARNING_RATE
        cfg.num_epochs = cfg_inst.NUM_EPOCHS
        cfg.minibatch = cfg_inst.MINIBATCH
        cfg.gamma = cfg_inst.GAE_GAMMA
        cfg.lambda_ = cfg_inst.GAE_LAMBDA
        if not is_selfplay:
            cfg.entropy_coef = cfg_inst.ENTROPY_COEF

        run_seed = random.randint(1, 0x7FFFFFFF) ^ (int(time.time()) & 0xFFFFFF)
        print(f"  base_seed     : {run_seed} (randomized)")
        trainer = puyotan_native.CppPPOTrainer(
            num_envs    = cfg_inst.NUM_ENVS,
            num_steps   = cfg_inst.STEPS_PER_ITER,
            arch        = "mlp" if arch == "light_mlp" else arch,
            hidden_dim  = cfg_inst.HIDDEN_DIM,
            base_seed   = run_seed,
            cfg         = cfg,
            arch_params = cfg_inst.ARCH_PARAMS,
        )

        reward_config_path = BASE_DIR / "native" / "resources" / config_name
        trainer.env.reward_calc.load_from_json(str(reward_config_path))

        # Set max episode steps (only applies to solo mode or if set)
        if not is_selfplay and cfg_inst.MAX_EPISODE_STEPS > 0:
            trainer.env.setMaxEpisodeSteps(cfg_inst.MAX_EPISODE_STEPS)
            print(f"  max_episode_steps : {cfg_inst.MAX_EPISODE_STEPS}")

        # Load checkpoint
        if latest_pt.exists():
            print(f"Resuming from checkpoint: {latest_pt}")
            trainer.load(str(latest_pt))
        elif is_selfplay:
            # For selfplay, if puyotan_latest.pt does not exist, try to load puyotan_solo_latest.pt to bootstrap
            solo_fallback = arch_dir / "puyotan_solo_latest.pt"
            if solo_fallback.exists():
                print(f"Initializing self-play with solo checkpoint: {solo_fallback}")
                trainer.load(str(solo_fallback))

        # ---------------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------------
        acc_loss = acc_sps = acc_avg_max = acc_reward = acc_score = 0.0
        acc_max_chain = 0
        acc_game_len = acc_max_potential = 0.0

        for i in range(cfg_inst.TOTAL_ITERS):
            iteration = i + 1

            # Opponent (P2) snapshot synchronization in self-play
            if is_selfplay and iteration % cfg_inst.SNAPSHOT_INTERVAL == 1:
                trainer.save(str(latest_pt))
                trainer.loadP2(str(latest_pt))

            t0 = time.perf_counter()

            # Opponent behavior flag
            # Solo: Always mirrored (P2 mirrors P1's actions)
            # Self-play: Random until first checkpoint is saved, then uses the trained policy loaded in P2
            if is_selfplay:
                p2_random = not latest_pt.exists()
            else:
                p2_random = False

            metrics = trainer.trainStep(p2_random=p2_random)

            elapsed = time.perf_counter() - t0
            sps     = (cfg_inst.NUM_ENVS * cfg_inst.STEPS_PER_ITER) / elapsed

            acc_loss      += metrics.loss
            acc_sps       += sps
            acc_avg_max   += metrics.avg_max_chain
            acc_reward    += metrics.avg_reward
            acc_score     += metrics.avg_game_score
            acc_max_chain  = max(acc_max_chain, metrics.max_chain)
            acc_game_len  += metrics.avg_game_len
            acc_max_potential += metrics.avg_max_potential

            if iteration % cfg_inst.LOG_INTERVAL == 0 or iteration == 1:
                div = min(iteration, cfg_inst.LOG_INTERVAL)
                print(
                    f"[Iter {iteration:4d}/{cfg_inst.TOTAL_ITERS}]"
                    f"  Loss={acc_loss/div:6.3f}"
                    f"  AvgRew={acc_reward/div:6.3f}"
                    f"  AvgScore={acc_score/div:6.1f}"
                    f"  AvgMax={acc_avg_max/div:4.2f}"
                    f"  Len={acc_game_len/div:5.1f}"
                    f"  Pot={acc_max_potential/div:4.2f}"
                    f"  Max={acc_max_chain:2d}"
                    f"  SPS={acc_sps/div:.0f}"
                )
                acc_loss = acc_sps = acc_max_chain = acc_avg_max = acc_reward = acc_score = 0.0
                acc_game_len = acc_max_potential = 0.0

            if iteration % cfg_inst.SAVE_INTERVAL == 0 or iteration == cfg_inst.TOTAL_ITERS:
                trainer.save(str(session_pt))
                shutil.copy2(str(session_pt), str(latest_pt))
                print(f"Saved checkpoint -> {session_pt.name} and {latest_pt.name}")

                # Automatically export to ONNX format (run in subprocess to prevent DLL conflicts)
                print("Exporting checkpoint to ONNX format...")
                hidden_dim = cfg_inst.HIDDEN_DIM
                channels = cfg_inst.ARCH_PARAMS.get("channels", 32 if arch == "resnet" else 24 if arch == "cnn" else 0)
                num_blocks = cfg_inst.ARCH_PARAMS.get("num_blocks", 2 if arch == "resnet" else 0)
                
                cmd = [
                    sys.executable,
                    "-m", "orchestrator.export_onnx",
                    "--arch", arch,
                    "--hidden_dim", str(hidden_dim),
                    "--channels", str(channels),
                    "--num_blocks", str(num_blocks),
                    "--pt_path", str(latest_pt),
                ]
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                
                try:
                    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to export to ONNX: {e}")
                    print(f"Stdout:\n{e.stdout}")
                    print(f"Stderr:\n{e.stderr}")
                except Exception as e:
                    print(f"Failed to launch ONNX export process: {e}")

        print("Training finished!")

    except Exception:
        print("CRITICAL ERROR in training_loop:")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuyotanAI Training System (C++ / LibTorch)")
    parser.add_argument("--mode",   type=str, default="solo", choices=["solo", "selfplay"],
                        help="Training mode: 'solo' (build chain building) or 'selfplay' (against freeze version of self)")
    parser.add_argument("--arch",   type=str, default="mlp", choices=["mlp", "cnn", "resnet", "light_mlp"],
                        help="Model backbone architecture (default: mlp)")
    parser.add_argument("--config", type=str, default=None,
                        help="Reward weights JSON file name (default: reward_solo.json for solo, reward_match.json for selfplay)")
    args = parser.parse_args()

    run_training(
        mode=args.mode,
        config_name=args.config,
        arch=args.arch,
    )
