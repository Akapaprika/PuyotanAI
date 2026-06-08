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
 * @struct BeamEvalWeights
 * @brief Tunable weights for the beam search evaluation function.
 *
 * These are deliberately separate from RewardWeights to allow
 * independent tuning of the lookahead heuristic vs. the RL reward signal.
 */
struct BeamEvalWeights {
    // Primary driver: potential chain score when one more puyo is added
    float potential_score_scale  = 1.0f;
    // Bonus per puyo that has >= 2 same-color neighbors (forms a group)
    float connectivity_bonus     = 0.4f;
    // Penalty per puyo with no same-color neighbors (isolated)
    float isolated_penalty       = -0.6f;
    // Penalty per puyo of any color that lies beneath an ojama
    float buried_penalty         = -1.5f;
    // Penalty proportional to height variance across columns
    float height_variance_penalty = -0.3f;
    // Penalty per unit of height in the death column (col 2)
    float death_col_penalty      = -1.0f;
    // Bonus per immediate chain fired this step (carry-through reward)
    float chain_bonus_per_step   = 2.0f;
    // Exponent applied to chain count (rewards big chains super-linearly)
    float chain_power            = 2.0f;
};

/**
 * @class BeamEvaluator
 * @brief Stateless board scorer for use inside beam search.
 *
 * All methods are static and branchless where possible, relying on the
 * same BitBoard SIMD primitives used by RewardCalculator.
 */
class BeamEvaluator {
  public:
    /**
     * @brief Evaluate a board state and return a heuristic score (higher = better).
     *
     * @param board   The board to evaluate.
     * @param w       Evaluation weights.
     * @param chain   Number of chains already fired to reach this state (0 if placement only).
     * @param score   Raw score delta from the chain (0 if no chain fired).
     */
    static float evaluate(const Board& board,
                          const BeamEvalWeights& w,
                          int chain = 0,
                          int score = 0,
                          bool calculate_potential = true) noexcept {
        float r = 0.0f;

        // --- Immediate chain reward ---
        if (chain > 0) {
            r += std::pow(static_cast<float>(chain), w.chain_power) * w.chain_bonus_per_step;
        }

        // --- Board metrics (BitBoard-level, branchless) ---
        int conn = 0, iso = 0;
        for (int c = 0; c < config::Rule::kColors; ++c) {
            const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
            if (bb.empty()) continue;

            const BitBoard U = bb.shiftUpRaw();
            const BitBoard D = bb.shiftDownRaw();
            const BitBoard L = bb.shiftLeftRaw();
            const BitBoard R = bb.shiftRightRaw();

            // Puyos with >= 2 same-color neighbors
            const BitBoard has2 = bb & ((U & D) | (L & R) | ((U | D) & (L | R)));
            conn += has2.popcount();

            // Isolated: no same-color neighbors
            BitBoard iso_bb = bb;
            iso_bb.andNot(U | D | L | R);
            iso += iso_bb.popcount();
        }

        r += static_cast<float>(conn)  * w.connectivity_bonus;
        r += static_cast<float>(iso)   * w.isolated_penalty;

        // --- Height variance ---
        {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int x = 0; x < config::Board::kWidth; ++x) {
                float h = static_cast<float>(board.getColumnHeight(x));
                sum    += h;
                sum_sq += h * h;
            }
            constexpr float n = static_cast<float>(config::Board::kWidth);
            float mean = sum / n;
            float var  = sum_sq / n - mean * mean;
            r += var * w.height_variance_penalty;
        }

        // --- Death column height ---
        r += static_cast<float>(board.getColumnHeight(config::Rule::kDeathCol))
             * w.death_col_penalty;

        // --- Buried puyo count (colored puyos beneath any ojama shadow) ---
        {
            const BitBoard& oj = board.getBitboard(Cell::Ojama);
            if (!oj.empty()) {
                // Smear ojama shadow downward via arithmetic right-shift
                BitBoard s = oj;
                s |= s.shiftDownRaw();
                s.lo |= _mm_srli_epi64(s.m128, 2).m128i_u64[0];
                s.hi |= _mm_srli_epi64(s.m128, 2).m128i_u64[1];
                s.lo |= _mm_srli_epi64(s.m128, 4).m128i_u64[0];
                s.hi |= _mm_srli_epi64(s.m128, 4).m128i_u64[1];
                s.lo |= _mm_srli_epi64(s.m128, 8).m128i_u64[0];
                s.hi |= _mm_srli_epi64(s.m128, 8).m128i_u64[1];

                const BitBoard all_colored =
                    board.getBitboard(Cell::Red)    | board.getBitboard(Cell::Blue) |
                    board.getBitboard(Cell::Green)  | board.getBitboard(Cell::Yellow);
                int buried = (all_colored & s).popcount();
                r += static_cast<float>(buried) * w.buried_penalty;
            }
        }

        // --- Potential chain score (max achievable by adding exactly one puyo) ---
        if (calculate_potential) {
            int max_pot_score = 0;
            for (int x = 0; x < config::Board::kWidth; ++x) {
                int h = board.getColumnHeight(x);
                if (h >= config::Board::kChainableRows) continue;

                for (int c = 0; c < config::Rule::kColors; ++c) {
                    const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));

                    // Pre-check adjacency: if no adjacent puyos of color c, it cannot erase (size will be 1)
                    bool has_adjacent = false;
                    if (x > 0 && bb.get(x - 1, h)) has_adjacent = true;
                    if (x < config::Board::kWidth - 1 && bb.get(x + 1, h)) has_adjacent = true;
                    if (h > 0 && bb.get(x, h - 1)) has_adjacent = true;

                    if (!has_adjacent) continue;

                    Board temp = board;
                    temp.dropNewPiece(x, h, static_cast<Cell>(c));

                    ErasureData ed = Chain::findGroups(temp, 1u << c);
                    if (ed.num_erased == 0) continue;

                    int pot_chain = 0, pot_score = 0;
                    while (ed.num_erased > 0) {
                        ++pot_chain;
                        pot_score += Scorer::calculateStepScore(ed, pot_chain);
                        Chain::applyErasure(temp, ed);
                        uint32_t fallen = Gravity::execute(temp);
                        ed = Chain::findGroups(temp, fallen);
                        if (pot_chain >= 10) { pot_score += 99999; break; }
                    }
                    if (pot_score > max_pot_score)
                        max_pot_score = pot_score;
                }
            }
            r += static_cast<float>(max_pot_score) * w.potential_score_scale;
        }

        return r;
    }
};

} // namespace puyotan::search
