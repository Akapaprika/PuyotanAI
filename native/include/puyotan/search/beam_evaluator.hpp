#pragma once

#include <cmath>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/search/eval_weights.hpp>
#include <puyotan/search/potential_score.hpp>

namespace puyotan::search {

/**
 * @class SoloBeamEvaluator
 * @brief Stateless board scorer for use inside solo beam search.
 */
class SoloBeamEvaluator {
  public:
    /**
     * @brief Evaluate a board state and return a heuristic score for Solo mode.
     */
    static int32_t evaluate(const Board& board,
                            const SoloBeamEvalWeights& w) noexcept {
        // --- Precompute all column heights once ---
        int heights[config::Board::kWidth];
        {
            const __m128i occ_m128 = board.getOccupied().m128;
            const uint64_t lo = static_cast<uint64_t>(_mm_cvtsi128_si64(occ_m128));
            const uint64_t hi = static_cast<uint64_t>(_mm_extract_epi64(occ_m128, 1));

            heights[0] = static_cast<int>(_mm_popcnt_u64((lo >>  0) & config::Board::kColMask));
            heights[1] = static_cast<int>(_mm_popcnt_u64((lo >> 16) & config::Board::kColMask));
            heights[2] = static_cast<int>(_mm_popcnt_u64((lo >> 32) & config::Board::kColMask));
            heights[3] = static_cast<int>(_mm_popcnt_u64((lo >> 48) & config::Board::kColMask));
            heights[4] = static_cast<int>(_mm_popcnt_u64((hi >>  0) & config::Board::kColMask));
            heights[5] = static_cast<int>(_mm_popcnt_u64((hi >> 16) & config::Board::kColMask));
        }

        // --- Potential chain score (shared implementation in potential_score.hpp) ---
        return computeMaxPotentialScore(board, heights) * w.potential_score_scale;
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
    static int32_t evaluate(const Board& board,
                            const VsBeamEvalWeights& w,
                            const VsEvalContext* ctx = nullptr) noexcept {
        int32_t r = 0;

        // --- Precompute all column heights once ---
        int heights[config::Board::kWidth];
        {
            const __m128i occ_m128 = board.getOccupied().m128;
            const uint64_t lo = static_cast<uint64_t>(_mm_cvtsi128_si64(occ_m128));
            const uint64_t hi = static_cast<uint64_t>(_mm_extract_epi64(occ_m128, 1));

            heights[0] = static_cast<int>(_mm_popcnt_u64((lo >>  0) & config::Board::kColMask));
            heights[1] = static_cast<int>(_mm_popcnt_u64((lo >> 16) & config::Board::kColMask));
            heights[2] = static_cast<int>(_mm_popcnt_u64((lo >> 32) & config::Board::kColMask));
            heights[3] = static_cast<int>(_mm_popcnt_u64((lo >> 48) & config::Board::kColMask));
            heights[4] = static_cast<int>(_mm_popcnt_u64((hi >>  0) & config::Board::kColMask));
            heights[5] = static_cast<int>(_mm_popcnt_u64((hi >> 16) & config::Board::kColMask));
        }

        // --- Board metrics (BitBoard-level, branchless) ---
        {
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

            r += conn * w.connectivity_bonus;
            r += iso * w.isolated_penalty;
        }

        // --- Buried puyo count (colored puyos beneath any ojama shadow) ---
        {
            const BitBoard& oj = board.getBitboard(Cell::Ojama);
            if (!oj.empty()) {
                __m128i s_reg = oj.m128;
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 1));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 2));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 4));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 8));

                __m128i all_colored = _mm_andnot_si128(oj.m128, board.getOccupied().m128);
                __m128i buried_reg = _mm_and_si128(all_colored, s_reg);

                uint64_t b_lo = _mm_cvtsi128_si64(buried_reg);
                uint64_t b_hi = _mm_extract_epi64(buried_reg, 1);
                int buried =
                    static_cast<int>(std::popcount(b_lo) + std::popcount(b_hi));
                r += buried * w.buried_penalty;
            }
        }

        // --- Potential chain score (shared implementation in potential_score.hpp) ---
        if constexpr (CalculatePotential) {
            r += computeMaxPotentialScore(board, heights) * w.potential_score_scale;
        }

        return r;
    }
};

} // namespace puyotan::search
