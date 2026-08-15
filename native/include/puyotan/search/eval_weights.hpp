#pragma once

#include <cstdint>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>

namespace puyotan::search {

/**
 * @struct SoloBeamEvalWeights
 * @brief Tunable weights for the solo beam search evaluation function.
 */
struct SoloBeamEvalWeights {
    float potential_score_scale = 1.0f;
};

/**
 * @struct VsBeamEvalWeights
 * @brief Tunable weights for the VS beam search evaluation function.
 */
struct VsBeamEvalWeights {
    float potential_score_scale = 1.0f;
    float connectivity_bonus    = 0.4f;
    float isolated_penalty      = -0.6f;
    float buried_penalty        = -1.5f;
    float fire_bias             = 1.0f;
    float incoming_ojama_penalty = -2.0f;

    // --- Dynamic Attack Search Bias Multipliers ---
    float incoming_threat_bias    = 1.5f;
    float counter_attack_bias     = 1.4f;
    float timing_advantage_bias   = 1.2f;

    // --- Dynamic Evaluation Parameters ---
    float urgency_weight            = 0.8f;
    float lethal_danger_scale       = 1.0f;
    float effective_strike_multiplier = 1.5f;
};

/**
 * @struct VsEvalContext
 * @brief Snapshot of the opponent's (and own) game state at the moment of an AI decision.
 */
struct VsEvalContext {
    Board      enemy_field;                              // enemy board snapshot
    int        enemy_active_next_pos = 0;               // enemy current tsumo index
    ActionType enemy_action_type = ActionType::None;    // current action state
    uint8_t    enemy_chain_count = 0;                   // resolved chain steps
    int        enemy_score       = 0;                   // cumulative score
    int        enemy_used_score  = 0;                   // score converted to ojama
    int        enemy_best_attack_score = 0;             // enemy best immediate attack score
    int        enemy_prepare_turns     = 99;            // enemy turns needed to fire best attack
    int        enemy_best_within_4     = 0;             // enemy best attack score within 4 turns
    int        my_best_within_4        = 0;             // my best attack score within 4 turns
    uint16_t   enemy_active_ojama     = 0;              // ojama ready to fall
    uint16_t   enemy_non_active_ojama = 0;              // ojama still cancelable
    uint16_t   my_active_ojama        = 0;              // my ojama ready to fall
    uint16_t   my_non_active_ojama    = 0;              // my ojama still cancelable
};

} // namespace puyotan::search
