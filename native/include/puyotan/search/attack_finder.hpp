#pragma once

#include <vector>
#include <puyotan/core/board.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan::search {

/**
 * @struct AttackCandidate
 * @brief Represents a single chain attack opportunity achievable within lookahead depth.
 */
struct AttackCandidate {
    int first_action = -1;  ///< Initial RL action index leading to this attack
    int score = 0;          ///< Total score of the chain
    int chain_count = 0;    ///< Number of steps (chains) in the attack
    int prepare_turns = 1;  ///< Turns (tsumo drops) needed before firing
    int total_frames = 1;   ///< Estimated total frames/turns to complete firing
    bool is_all_clear = false; ///< True if field is cleared after chain
};

/**
 * @brief Collects all potential chain attack opportunities up to max_depth tsumo steps.
 * 
 * @param field The starting board state.
 * @param tsumo The tsumo generator.
 * @param tsumo_base Starting index into the tsumo sequence.
 * @param max_depth Number of tsumo steps to search ahead (default: 3).
 * @return std::vector<AttackCandidate> Sorted list of attack opportunities (highest score first).
 */
std::vector<AttackCandidate> collectAttackCandidates(
    const Board& field,
    const TsumoSequence& tsumo_seq,
    int tsumo_base,
    int max_depth = 3,
    int max_states_per_layer = 250
) noexcept;

} // namespace puyotan::search
