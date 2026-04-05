"""
Centralized training configuration for PuyotanAI.
Enables switching between experiment modes (Lightweight, Production) in one place.
"""

# ---------------------------------------------------------------------------
# Hardware-specific (AMD 3020e / 2-core) "Lightweight" Profile
# ---------------------------------------------------------------------------
# Set these to higher values (e.g., 256/128/8192) for high-performance batch training.

NUM_ENVS       = 64
STEPS_PER_ITER = 32
TOTAL_ITERS    = 1000

# Logging and Checkpoints
LOG_INTERVAL      = 1
SAVE_INTERVAL     = 50
SNAPSHOT_INTERVAL = 50   # Used in self-play

# Model Architecture
HIDDEN_DIM = 128

# PPO Hyperparameters
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 4
MINIBATCH     = 1024
GAE_GAMMA     = 0.99
GAE_LAMBDA    = 0.95
