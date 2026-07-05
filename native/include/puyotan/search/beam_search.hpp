#pragma once

#include <utility>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @struct SoloBeamConfig
 * @brief Parameters controlling solo beam search behaviour.
 */
struct SoloBeamConfig {
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    SoloBeamEvalWeights eval_weights;
};

/**
 * @struct VsBeamConfig
 * @brief Parameters controlling VS beam search behaviour.
 */
struct VsBeamConfig {
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    VsBeamEvalWeights eval_weights;
};

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for Solo mode.
 */
std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo&         tsumo,
                                     const SoloBeamConfig& cfg) noexcept;

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for VS mode.
 */
std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const Tsumo&         tsumo,
                                   const VsBeamConfig&  cfg) noexcept;

} // namespace puyotan::search
