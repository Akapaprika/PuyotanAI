"""
Centralized PPO hyperparameters and training defaults.

Keeping all magic numbers in one place allows easy tuning without
hunting through implementation files.
"""

# ---------------------------------------------------------------------------
# PPO algorithm hyperparameters
# ---------------------------------------------------------------------------
GAMMA           = 0.99   # Discount factor
LAMBDA          = 0.95   # GAE lambda (bias-variance trade-off)
CLIP_EPS        = 0.2    # PPO clipping epsilon
ENTROPY_COEF    = 0.05   # Entropy bonus coefficient (exploration)
VALUE_LOSS_COEF = 0.5    # Critic loss weight
MAX_GRAD_NORM   = 0.5    # Gradient clipping threshold
NUM_EPOCHS      = 2      # PPO update passes per rollout
MINIBATCH_SIZE  = 2048   # Samples per gradient step
LEARNING_RATE   = 3e-4   # Adam learning rate

# ---------------------------------------------------------------------------
# Rollout defaults (can be overridden per-orchestrator)
# ---------------------------------------------------------------------------
DEFAULT_NUM_ENVS        = 256
DEFAULT_STEPS_PER_ITER  = 128
DEFAULT_HIDDEN_DIM      = 128
