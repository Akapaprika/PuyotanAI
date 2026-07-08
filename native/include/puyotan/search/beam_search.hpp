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
 * @struct VsEvalContext
 * @brief Snapshot of the opponent's (and own) game state at the moment of
 *        an AI decision. Populate this before every call to vsBeamSearch().
 *
 * When enemy_action_type is ActionType::Chain or ChainFall the enemy is
 * mid-chain: enemy_field holds the board snapshot at that point, and
 * enemy_chain_count reflects the number of chain steps already resolved.
 * Callers can run Chain::scanGroups() on enemy_field to estimate remaining
 * chain score without waiting for the chain to fully resolve.
 */
struct VsEvalContext {
    Board      enemy_field;                              // enemy board snapshot
    ActionType enemy_action_type = ActionType::None;    // current action state
    uint8_t    enemy_chain_count = 0;                   // resolved chain steps
    int        enemy_score       = 0;                   // cumulative score
    int        enemy_used_score  = 0;                   // score converted to ojama
    uint16_t   enemy_active_ojama     = 0;              // ojama ready to fall
    uint16_t   enemy_non_active_ojama = 0;              // ojama still cancelable
    uint16_t   my_active_ojama        = 0;              // my ojama ready to fall
    uint16_t   my_non_active_ojama    = 0;              // my ojama still cancelable
};

/**
 * @struct VsBeamConfig
 * @brief Parameters controlling VS beam search behaviour.
 *
 * Populate context from the live match state before every call to
 * vsBeamSearch(). The evaluator does not yet use context fields - this is
 * preparation wiring for future metrics.
 */
struct VsBeamConfig {
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    VsBeamEvalWeights eval_weights;
    VsEvalContext     context;                          // set before each call
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
