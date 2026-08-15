#pragma once

#include <utility>
#include <vector>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for Solo mode.
 */
std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo&         tsumo,
                                     const SoloBeamConfig& cfg,
                                     BeamSearchSession*   session = nullptr) noexcept;

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for VS mode.
 */
std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const Tsumo&         tsumo,
                                   const VsBeamConfig&  cfg,
                                   BeamSearchSession*   session = nullptr) noexcept;

/**
 * @brief Runs a VS beam search and returns the top N unique candidate actions with their scores.
 */
std::vector<std::pair<int, float>> vsBeamSearchTopN(const PuyotanPlayer& player,
                                                    const Tsumo&         tsumo,
                                                    const VsBeamConfig&  cfg,
                                                    int                  top_n = 5) noexcept;

} // namespace puyotan::search
