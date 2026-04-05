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
//   [17 .. 21]  Left,  col 0-4   (5 actions  sub puyo at col-1, but col 0
//                                  maps sub to col -1 which the engine
//                                  accepts as a pass-through; see note below)
//
// NOTE: Indices 17 (Left, col=0) is geometrically marginal but kept as-is
// to preserve compatibility with all trained ONNX models.
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
    return Action{ActionType::Put, static_cast<int8_t>(idx - (3 * w - 1)), Rotation::Left};
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
    void stepNative(std::span<const int> p1_actions,
                    std::span<const int> p2_actions,
                    std::span<float> out_rewards,
                    std::span<float> out_dones,
                    std::span<int32_t> out_chains,
                    std::span<int32_t> out_scores,
                    std::span<uint8_t> out_obs);

    /**
     * @brief Write observations to a pre-allocated uint8 buffer (no Python objects).
     * @param out_obs Buffer of size [n * kBytesPerObservation].
     */
    void getObservationsNative(std::span<uint8_t> out_obs) const;

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

  private:
    std::vector<PuyotanMatch> matches_;
    uint32_t base_seed_;
};
} // namespace puyotan
