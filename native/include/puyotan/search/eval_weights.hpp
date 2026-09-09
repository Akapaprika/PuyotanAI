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

/**
 * @struct MatchBeamEvalWeights
 * @brief Tunable weights for the Match-simulation beam search evaluation function.
 *
 * 評価式（自分目線での優位スコア）:
 *   score = logPot(me) - logPot(enemy)          // 対数量子化ポテンシャル差
 *         + w_board * (boardQuality(me) - boardQuality(enemy))  // 盤面形質差
 *         - w_ojama_active  * C * (me.active_ojama^2)           // 落下確定おじゃま 二乗ペナルティ
 *         - w_ojama_pending * me.non_active_ojama               // 保留おじゃまペナルティ
 *         + w_ojama_active  * C * (enemy.active_ojama^2)        // 相手への同等ボーナス
 *         + w_ojama_pending * enemy.non_active_ojama            // 相手の保留ボーナス
 *         - w_height * dangerHeight(me)                         // 盤面高さ危険ペナルティ
 */
struct MatchBeamEvalWeights {
    // --- Potential score (log-quantized) ---
    int32_t potential_score_scale   = 1;   ///< ポテンシャルスコアのスケール係数
    int32_t log_pot_base_permille   = 1000; ///< log量子化の底 (1000 = log base 自然, 実際はint近似)

    // --- Board quality (self - enemy) ---
    int32_t connectivity_bonus      = 15;   ///< 連結ぷよボーナス (per puyo with >=2 neighbors)
    int32_t isolated_penalty        = -30;  ///< 孤立ぷよペナルティ (per isolated puyo)
    int32_t buried_penalty          = -80;  ///< おじゃま下の埋没ペナルティ (per buried colored puyo)

    // --- Ojama pressure (quadratic for active, linear for pending) ---
    int32_t active_ojama_coeff      = 70;   ///< C: 落下確定おじゃま二乗係数 (score = -C * n^2)
    int32_t pending_ojama_penalty   = 40;   ///< 保留おじゃまペナルティ (per ojama)

    // --- Height danger (my field) ---
    int32_t height_danger_threshold = 10;   ///< 危険とみなす最低高さ (0-indexed, 10 = row 11)
    int32_t height_danger_penalty   = -300; ///< 危険列1本あたりのペナルティ

    // --- Terminal state scores ---
    int32_t win_score               = 10000000;  ///< 勝利確定スコア
    int32_t draw_score              = -5000000;  ///< 引き分けスコア（やや不利）
};

} // namespace puyotan::search
