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
    int32_t potential_score_scale = 1;
};

/**
 * @struct VsBeamEvalWeights
 * @brief Tunable weights for the VS beam search evaluation function (integer scale).
 */
struct VsBeamEvalWeights {
    int32_t potential_score_scale      = 1;
    int32_t connectivity_bonus         = 20;     ///< Score per connected puyo pair (+20 pts)
    int32_t isolated_penalty           = -40;    ///< Penalty per isolated puyo (-40 pts)
    int32_t buried_penalty             = -100;   ///< Penalty per colored puyo buried under ojama (-100 pts)
    int32_t fire_bias_permille         = 1000;   ///< Immediate fire multiplier permille (1000 = 1.0x)
    int32_t incoming_ojama_penalty     = -140;   ///< Penalty per incoming ojama (-140 pts)

    // --- Dynamic Attack Search Bias Multipliers (Permille: 1000 = 1.0x) ---
    int32_t incoming_threat_bias_permille    = 1500;
    int32_t counter_attack_bias_permille     = 1400;
    int32_t timing_advantage_bias_permille   = 1200;

    // --- Dynamic Evaluation Parameters ---
    int32_t urgency_weight_permille            = 800;
    int32_t lethal_danger_scale                = 1;
    int32_t effective_strike_multiplier_permille = 1500;
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
