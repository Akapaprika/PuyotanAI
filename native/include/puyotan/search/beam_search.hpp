#pragma once

#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @struct BeamConfig
 * @brief Parameters controlling beam search behaviour.
 */
struct BeamConfig {
    /// Number of top candidate boards retained at each depth level.
    int beam_width = 500;
    /// Number of tsumo pieces to look ahead (depth of the search tree).
    /// Uses the shared Tsumo sequence starting from the player's active_next_pos.
    int look_ahead = 3;
    /// Evaluation weights applied at every leaf node and intermediate node.
    BeamEvalWeights eval_weights;
};

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index.
 *
 * The search expands all 22 RL actions at each depth level, simulates the
 * resulting board state (drop + chain resolution + gravity), evaluates it,
 * and retains the top `cfg.beam_width` candidates to continue exploring.
 *
 * At depth 0 the action that leads to the highest-scored subtree is returned.
 *
 * @param player  Current player state (field + tsumo position).
 * @param tsumo   Shared tsumo generator for the match.
 * @param cfg     Beam search configuration (width, depth, weights).
 * @return        RL action index in [0, kNumRLActions). Returns 0 on error.
 */
int beamSearch(const PuyotanPlayer& player,
               const Tsumo&         tsumo,
               const BeamConfig&    cfg) noexcept;

} // namespace puyotan::search
