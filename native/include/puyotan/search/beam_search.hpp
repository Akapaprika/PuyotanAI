#pragma once

#include <utility>
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
    /// Diverse Beam Search (DBS) parameters.
    int dbs_max_similar = 0;
    /// Evaluation weights applied at every leaf node and intermediate node.
    BeamEvalWeights eval_weights;
};

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score.
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
 * @return        std::pair of RL action index and its expected score.
 */
std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo&         tsumo,
                                     const BeamConfig&    cfg) noexcept;

std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const Tsumo&         tsumo,
                                   const BeamConfig&    cfg) noexcept;

} // namespace puyotan::search
