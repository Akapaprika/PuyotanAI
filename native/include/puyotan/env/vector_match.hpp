#pragma once

#include <optional>
#include <span>
#include <vector>

#include <puyotan/engine/match.hpp>
#include <puyotan/env/reward.hpp>

namespace puyotan {
// -----------------------------------------------------------------------
// RL Action Table
//
// Maps a flat action index [0, kNumRLActions) to a concrete (column, rotation)
// pair. This is the SINGLE SOURCE OF TRUTH for the action space used by
// both the C++ training loop and the Python inference / GUI.
//
// Layout (w = Board::kWidth = 6):
//   [ 0 ..  5]  Up,    col 0-5   (6 actions)
//   [ 6 .. 10]  Right, col 0-4   (5 actions  sub puyo at col+1)
//   [11 .. 16]  Down,  col 0-5   (6 actions)
//   [17 .. 21]  Left,  col 1-5   (5 actions  sub puyo at col-1)
//
// NOTE: All 22 actions are now strictly within board boundaries.
// -----------------------------------------------------------------------
inline constexpr int kNumRLActions =
    config::Board::kWidth + (config::Board::kWidth - 1) +
    config::Board::kWidth + (config::Board::kWidth - 1); // = 22

/// Convert a flat RL action index to a concrete Action.
/// Returns ActionType::Pass for out-of-range indices.
[[nodiscard]] inline Action getRLAction(int idx) noexcept {
    constexpr int w = config::Board::kWidth;
    if (idx < 0 || idx >= kNumRLActions)
        return Action{ActionType::Pass};
    if (idx < w)
        return Action{ActionType::Put, static_cast<int8_t>(idx), Rotation::Up};
    if (idx < w + (w - 1))
        return Action{ActionType::Put, static_cast<int8_t>(idx - w), Rotation::Right};
    if (idx < w + (w - 1) + w)
        return Action{ActionType::Put, static_cast<int8_t>(idx - (2 * w - 1)), Rotation::Down};
    
    // Left rotation: col 1 to 5 (idx 17..21 maps to col 1..5)
    return Action{ActionType::Put, static_cast<int8_t>(idx - (3 * w - 1) + 1), Rotation::Left};
}

/**
 * @class PuyotanVectorMatch
 * @brief Synchronous parallel orchestrator for batch Puyo Puyo matches.
 */
class PuyotanVectorMatch {
  public:
    explicit PuyotanVectorMatch(int num_matches, uint32_t base_seed = 1u);

    void reset(int id = -1) noexcept;
    std::vector<int> stepUntilDecision();
    void setActions(const std::vector<int>& match_indices,
                    const std::vector<int>& player_ids,
                    const std::vector<Action>& actions);

    /**
     * @brief High-performance native step for the C++ training loop.
     *
     * Bypasses all pybind11 overhead (no GIL, no Python object creation).
     * All output buffers must be pre-allocated by the caller.
     *
     * @param p1_actions  Action indices for P1 [n]
     * @param p2_actions  Action indices for P2 [n]
     * @param out_rewards Output float32 rewards [n]
     * @param out_dones   Output float32 terminated flags [n]
     * @param out_chains  Output int32 chain counts [n]
     * @param out_scores  Output int32 delta scores [n]
     * @param out_obs     Output uint8 observations [n * obs_bytes]
     */
    void stepNative(std::span<const int8_t> p1_actions,
                    std::span<const int8_t> p2_actions,
                    std::span<float> out_rewards,
                    std::span<float> out_dones,
                    std::span<int8_t> out_chains,
                    std::span<int32_t> out_scores,
                    std::span<int8_t> out_potentials,
                    std::span<uint8_t> out_obs) noexcept;

    /**
     * @brief Write observations to a pre-allocated uint8 buffer (no Python objects).
     * @param out_obs Buffer of size [n * kBytesPerObservation].
     */
    void getObservationsNative(std::span<uint8_t> out_obs) const noexcept;

    size_t size() const {
        return matches_.size();
    }
    PuyotanMatch& getMatch(int i) {
        return matches_[i];
    }
    const PuyotanMatch& getMatch(int i) const {
        return matches_[i];
    }

    RewardCalculator reward_calc;

    /// Set maximum steps per episode (0 = unlimited).
    /// When an episode exceeds this limit it is force-reset with done=true.
    void setMaxEpisodeSteps(int n) noexcept { max_episode_steps_ = n; }
    int getMaxEpisodeSteps() const noexcept { return max_episode_steps_; }

  private:
    std::vector<PuyotanMatch> matches_;
    std::vector<uint32_t> env_seeds_;
    uint32_t base_seed_;
    int max_episode_steps_ = 0;          // 0 = unlimited
    std::vector<int> episode_steps_;     // per-env step counter
};
} // namespace puyotan
