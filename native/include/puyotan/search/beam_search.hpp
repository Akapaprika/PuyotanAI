#pragma once

#include <utility>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for Solo mode.
 */
std::pair<int, int32_t> soloBeamSearch(const PuyotanPlayer& player,
                                       const Tsumo&         tsumo,
                                       const SoloBeamConfig& cfg,
                                       BeamSearchSession*   session = nullptr) noexcept;

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for VS mode.
 */
std::pair<int, int32_t> vsBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo&         tsumo,
                                     const VsBeamConfig&  cfg,
                                     BeamSearchSession*   session = nullptr) noexcept;

} // namespace puyotan::search
