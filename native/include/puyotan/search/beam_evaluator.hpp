#pragma once

#include <cmath>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

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
    float fire_bias               = 1.0f;
    float incoming_ojama_penalty  = -2.0f;
    float attack_advantage_bonus  = 5.0f;

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

// ---------------------------------------------------------------------------
// Shared helper: compute the maximum potential chain score for a board.
// Used by both SoloBeamEvaluator and VsBeamEvaluator to avoid code duplication.
// ---------------------------------------------------------------------------
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board, const int heights[config::Board::kWidth]) noexcept {
    int max_pot_score = 0;
    for (int x = 0; x < config::Board::kWidth; ++x) {
        const int h = heights[x];
        if (h >= config::Board::kChainableRows)
            continue;

        for (int c = 0; c < config::Rule::kColors; ++c) {
            const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
            const BitBoard neighbor =
                bb.shiftUpRaw() | bb.shiftDownRaw() |
                bb.shiftLeftRaw() | bb.shiftRightRaw();
            if (!neighbor.get(x, h))
                continue;

            Board temp = board;
            temp.dropNewPiece(x, h, static_cast<Cell>(c));

            ErasureData ed;
            Chain::scanGroups(temp, ed, 1u << c);
            if (ed.num_erased == 0)
                continue;

            int pot_chain = 0, pot_score = 0;
            while (ed.num_erased > 0) {
                ++pot_chain;
                pot_score += Scorer::calculateStepScore(ed, pot_chain);
                Chain::applyErasure(temp, ed);
                uint32_t fallen = Gravity::execute(temp);
                Chain::scanGroups(temp, ed, fallen);
            }
            max_pot_score = (pot_score > max_pot_score) ? pot_score : max_pot_score;
        }
    }
    return max_pot_score;
}

/**
 * @class SoloBeamEvaluator
 * @brief Stateless board scorer for use inside solo beam search.
 */
class SoloBeamEvaluator {
  public:
    template <bool CalculatePotential = true>
    static float evaluate(const Board& board,
                          const SoloBeamEvalWeights& w) noexcept {
        float r = 0.0f;
        if constexpr (CalculatePotential) {
            const BitBoard& occ = board.getOccupied();
            int heights[config::Board::kWidth];
            heights[0] = static_cast<int>(_mm_popcnt_u64((occ.lo >> 0) & 0xFFFFu));
            heights[1] = static_cast<int>(_mm_popcnt_u64((occ.lo >> 16) & 0xFFFFu));
            heights[2] = static_cast<int>(_mm_popcnt_u64((occ.lo >> 32) & 0xFFFFu));
            heights[3] = static_cast<int>(_mm_popcnt_u64((occ.lo >> 48) & 0xFFFFu));
            heights[4] = static_cast<int>(_mm_popcnt_u64((occ.hi >> 0) & 0xFFFFu));
            heights[5] = static_cast<int>(_mm_popcnt_u64((occ.hi >> 16) & 0xFFFFu));
            r += static_cast<float>(computeMaxPotentialScore(board, heights)) * w.potential_score_scale;
        }
        return r;
    }
};

/**
 * @class VsBeamEvaluator
 * @brief Stateless board scorer for use inside VS beam search.
 */
class VsBeamEvaluator {
  public:
    /**
     * @brief Evaluate a board state and return a heuristic score for VS mode.
     */
    template <bool CalculatePotential = true>
    static float evaluate(const Board& board,
                          const VsBeamEvalWeights& w,
                          const VsEvalContext* ctx = nullptr) noexcept {
        float r = 0.0f;

        // --- Precompute column heights ---
        int heights[config::Board::kWidth];
        {
            const uint64_t lo = board.getOccupied().lo;
            const uint64_t hi = board.getOccupied().hi;
            heights[0] = static_cast<int>(_mm_popcnt_u64((lo >> 0) & 0xFFFFu));
            heights[1] = static_cast<int>(_mm_popcnt_u64((lo >> 16) & 0xFFFFu));
            heights[2] = static_cast<int>(_mm_popcnt_u64((lo >> 32) & 0xFFFFu));
            heights[3] = static_cast<int>(_mm_popcnt_u64((lo >> 48) & 0xFFFFu));
            heights[4] = static_cast<int>(_mm_popcnt_u64((hi >> 0) & 0xFFFFu));
            heights[5] = static_cast<int>(_mm_popcnt_u64((hi >> 16) & 0xFFFFu));
        }

        // --- Context Awareness & Block E: Lethal Danger Penalty ---
        // heights[] is already computed above; sum them instead of calling getOccupied() again.
        const int my_occupied = heights[0]+heights[1]+heights[2]+heights[3]+heights[4]+heights[5];
        if (ctx != nullptr) {
            const int total_incoming = ctx->my_active_ojama + ctx->my_non_active_ojama;
            if (total_incoming > 0) {
                r += static_cast<float>(total_incoming) * w.incoming_ojama_penalty;
            }

            // Block E: Lethal Danger Penalty (Suffocation Check)
            const int my_open_cells = 72 - my_occupied - total_incoming;
            const int enemy_ojama_potential = ctx->enemy_best_attack_score / config::Score::kTargetScore;

            if (enemy_ojama_potential >= my_open_cells && my_open_cells > 0) {
                r -= 185500.0f * w.lethal_danger_scale;
            }
        }

        // --- Potential chain score with Block C (Urgency Scale) ---
        if constexpr (CalculatePotential) {
            const int max_pot_score = computeMaxPotentialScore(board, heights);

            // --- Block C: Dynamic Urgency Scale ---
            // Compress potential_score_scale when enemy is close to firing,
            // promoting immediate sub-chains and counter-attacks to top of beam.
            float effective_pot_scale = w.potential_score_scale;
            if (ctx != nullptr && ctx->enemy_prepare_turns < 99) {
                const float urgency = std::min(1.0f, 1.0f / static_cast<float>(std::max(1, ctx->enemy_prepare_turns)));
                effective_pot_scale *= (1.0f - urgency * w.urgency_weight);
            }

            r += static_cast<float>(max_pot_score) * effective_pot_scale;
        }

        return r;
    }
};

} // namespace puyotan::search
