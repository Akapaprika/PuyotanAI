#pragma once

#include <cstdint>

namespace puyotan::rl {

// ---------------------------------------------------------------------------
// Observation dimensions — must match training/model.py exactly
// ---------------------------------------------------------------------------
inline constexpr int kNumActions   = 22;
inline constexpr int kObsPlayers   = 2;
inline constexpr int kObsColors    = 5;
inline constexpr int kObsCols      = 6;
inline constexpr int kObsRows      = 14;
inline constexpr int kObsFlatSize = kObsPlayers * kObsColors * kObsCols * kObsRows; // 840

// CNN: treat (players * colors) as channels
inline constexpr int kCnnInChannels = kObsPlayers * kObsColors; // 10

// ---------------------------------------------------------------------------
// Default PPO hyperparameters — must match training/config.py
// ---------------------------------------------------------------------------
inline constexpr float kDefaultGamma         = 0.99f;
inline constexpr float kDefaultLambda        = 0.95f;
inline constexpr float kDefaultClipEps      = 0.2f;
inline constexpr float kDefaultEntropyCoef  = 0.05f;
inline constexpr float kDefaultValueCoef    = 0.5f;
inline constexpr float kDefaultMaxGradNorm = 0.5f;
inline constexpr float kDefaultLr            = 3e-4f;
inline constexpr int   kDefaultNumEpochs    = 2;
inline constexpr int   kDefaultMinibatch     = 2048;

// ---------------------------------------------------------------------------
// Default rollout / environment defaults
// ---------------------------------------------------------------------------
inline constexpr int kDefaultNumEnvs  = 256;
inline constexpr int kDefaultNumSteps = 128;
inline constexpr int kDefaultHidden    = 128;

} // namespace puyotan::rl
