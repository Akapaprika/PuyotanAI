#pragma once

#include <utility>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_config.hpp>

namespace puyotan::search {

/**
 * @brief Runs a Match-simulation-based beam search.
 * Simulates full PuyotanMatch state transitions, simulating 22 actions for enemy
 * and look_ahead plies for the self player.
 * 
 * @param match The current match state
 * @param my_id The player ID of the searching AI (0 or 1)
 * @param cfg The search configuration
 * @return std::pair<int, int32_t> (best first RL action index, expected score)
 */
std::pair<int, int32_t> matchBeamSearch(const PuyotanMatch& match,
                                        int my_id,
                                        const MatchBeamConfig& cfg) noexcept;

} // namespace puyotan::search
