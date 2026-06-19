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
    float potential_score_scale = 1.0f;
    // Bonus per puyo that has >= 2 same-color neighbors (forms a group)
    float connectivity_bonus = 0.4f;
    // Penalty per puyo with no same-color neighbors (isolated)
    float isolated_penalty = -0.6f;
    // Penalty per puyo of any color that lies beneath an ojama
    float buried_penalty = -1.5f;
    // Multiplier applied to the best immediate fire score when comparing
    // fire-now vs. beam-continuation at depth 0.
    // fire_bias > 1.0  => prefer firing earlier
    // fire_bias == 1.0 => fire only if it strictly beats the beam score
    // fire_bias < 1.0  => prefer building (fire must be clearly better)
    float fire_bias = 1.0f;
    // Use fast approximate potential calculation (flood-fill).
    // NOTE: Disabling this gives more accurate multi-chain evaluation.
    // Fast mode only counts connected group size and cannot detect multi-step
    // chains.
    bool use_fast_potential = false;
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
    static __forceinline int fastGroupSize(const BitBoard& bb, int x,
                                           int h) noexcept {
        BitBoard bb_plus = bb;
        bb_plus.set(x, h);

        BitBoard component{};
        component.set(x, h);

        for (;;) {
            BitBoard next = component | (component.shiftUpRaw() & bb_plus) |
                            (component.shiftDownRaw() & bb_plus) |
                            (component.shiftRightRaw() & bb_plus) |
                            (component.shiftLeftRaw() & bb_plus);
            if (next == component)
                break;
            component = next;
        }
        return component.popcount();
    }

    /**
     * @brief Evaluate a board state and return a heuristic score (higher =
     * better).
     *
     * @tparam CalculatePotential  Compile-time flag: false skips the expensive
     * potential pass. Use evaluate<false>() from Expectimax inner loops.
     * @param board   The board to evaluate.
     * @param w       Evaluation weights.
     *
     * Note: Immediate fire score is handled at the beam search level (see
     * beamSearchImpl). This function evaluates only the board structure and
     * potential.
     */
    template <bool CalculatePotential = true, bool UseFastPotential = false,
              bool HasOjama = true>
    static float evaluate(const Board& board,
                          const BeamEvalWeights& w) noexcept {
        float r = 0.0f;

        // --- Precompute all column heights once ---
        // getColumnHeight uses _mm_popcnt_u32 per call. Caching here avoids
        // 13+ redundant popcnt calls scattered across variance, death col, and
        // potential.
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

        // --- Board metrics (BitBoard-level, branchless) ---
        int conn = 0, iso = 0;
        for (int c = 0; c < config::Rule::kColors; ++c) {
            const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
            if (bb.empty())
                continue;

            const BitBoard U = bb.shiftUpRaw();
            const BitBoard D = bb.shiftDownRaw();
            const BitBoard L = bb.shiftLeftRaw();
            const BitBoard R = bb.shiftRightRaw();

            const BitBoard UD = U | D;
            const BitBoard LR = L | R;

            // Puyos with >= 2 same-color neighbors
            const BitBoard has2 =
                bb & ((U & D) | (L & R) | (UD & LR));
            conn += has2.popcount();

            // Isolated: no same-color neighbors
            BitBoard iso_bb = bb;
            iso_bb.andNot(UD | LR);
            iso += iso_bb.popcount();
        }

        r += static_cast<float>(conn) * w.connectivity_bonus;
        r += static_cast<float>(iso) * w.isolated_penalty;

        // --- Buried puyo count (colored puyos beneath any ojama shadow) ---
        if constexpr (HasOjama) {
            const BitBoard& oj = board.getBitboard(Cell::Ojama);
            if (!oj.empty()) {
                // In-register SIMD downward shadow smearing (eliminates STLF
                // stalls)
                __m128i s_reg = oj.m128;
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 1));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 2));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 4));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 8));

                // Combine all colored boards using parallel register ANDNOT (Occupied & ~Ojama)
                __m128i all_colored = _mm_andnot_si128(oj.m128, board.getOccupied().m128);

                __m128i buried_reg = _mm_and_si128(all_colored, s_reg);

                // Direct register extraction to avoid any memory-store
                // penalties
                uint64_t b_lo = _mm_cvtsi128_si64(buried_reg);
                uint64_t b_hi = _mm_extract_epi64(buried_reg, 1);
                int buried =
                    static_cast<int>(std::popcount(b_lo) + std::popcount(b_hi));
                r += static_cast<float>(buried) * w.buried_penalty;
            }
        }

        // --- Potential chain score ---
        if constexpr (CalculatePotential) {
            if constexpr (UseFastPotential) {
                int max_gs = 0;
                for (int x = 0; x < config::Board::kWidth; ++x) {
                    const int h = heights[x]; // cached
                    if (h >= config::Board::kChainableRows)
                        continue;

                    for (int c = 0; c < config::Rule::kColors; ++c) {
                        const BitBoard& bb =
                            board.getBitboard(static_cast<Cell>(c));
                        // SIMD neighbor mask: 4 shift ops replace 3 conditional
                        // bb.get() calls. No per-bit boundary checks needed.
                        const BitBoard neighbor =
                            bb.shiftUpRaw() | bb.shiftDownRaw() |
                            bb.shiftLeftRaw() | bb.shiftRightRaw();
                        if (!neighbor.get(x, h))
                            continue;

                        int gs = fastGroupSize(bb, x, h);
                        max_gs = (gs > max_gs) ? gs : max_gs;
                    }
                }
                if (max_gs >= 4) {
                    float approx =
                        static_cast<float>(10 * max_gs * (max_gs - 3));
                    r += approx * w.potential_score_scale;
                }
            } else {
                int max_pot_score = 0;
                for (int x = 0; x < config::Board::kWidth; ++x) {
                    const int h = heights[x]; // cached
                    if (h >= config::Board::kChainableRows)
                        continue;

                    for (int c = 0; c < config::Rule::kColors; ++c) {
                        const BitBoard& bb =
                            board.getBitboard(static_cast<Cell>(c));
                        // SIMD neighbor mask: 4 shifts replace 3 conditional
                        // bb.get() calls.
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
                            pot_score +=
                                Scorer::calculateStepScore(ed, pot_chain);
                            Chain::applyErasure(temp, ed);
                            uint32_t fallen = Gravity::execute(temp);
                            Chain::scanGroups(temp, ed, fallen);
                        }
                        max_pot_score = (pot_score > max_pot_score)
                                            ? pot_score
                                            : max_pot_score;
                    }
                }
                r +=
                    static_cast<float>(max_pot_score) * w.potential_score_scale;
            }
        }

        return r;
    }
};

} // namespace puyotan::search
