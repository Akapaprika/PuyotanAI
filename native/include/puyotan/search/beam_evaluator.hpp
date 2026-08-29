#pragma once

#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
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
                            const SoloBeamEvalWeights& w,
                            uint32_t packed_heights) noexcept {
        return computeMaxPotentialScore(board, packed_heights) * w.potential_score_scale;
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
                            uint32_t packed_heights,
                            const VsEvalContext* ctx = nullptr) noexcept {
        int32_t r = 0;

        // --- Board metrics (BitBoard-level, branchless & 遅延 popcount) ---
        {
            __m128i all_has2 = _mm_setzero_si128();
            __m128i all_iso  = _mm_setzero_si128();

            for (int c = 0; c < config::Rule::kColors; ++c) {
                const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
                if (bb.empty())
                    continue;

                const __m128i bbm = bb.m128;
                const __m128i U = _mm_slli_epi64(bbm, 1);
                const __m128i D = _mm_srli_epi64(bbm, 1);
                const __m128i L = _mm_srli_si128(bbm, 2);
                const __m128i R = _mm_slli_si128(bbm, 2);

                const __m128i UD = _mm_or_si128(U, D);
                const __m128i LR = _mm_or_si128(L, R);

                // Puyos with >= 2 same-color neighbors: (U & D) | (L & R) | (UD & LR)
                const __m128i has2 = _mm_and_si128(bbm,
                    _mm_or_si128(_mm_or_si128(_mm_and_si128(U, D), _mm_and_si128(L, R)),
                                 _mm_and_si128(UD, LR)));
                all_has2 = _mm_or_si128(all_has2, has2);

                // Isolated: no same-color neighbors (bb & ~(UD | LR))
                const __m128i iso_m = _mm_andnot_si128(_mm_or_si128(UD, LR), bbm);
                all_iso = _mm_or_si128(all_iso, iso_m);
            }

            // ★ 4色合算後に 1 回だけ popcount を実行 (16命令 → 4命令に削減)
            const BitBoard b_has2(all_has2);
            const BitBoard b_iso(all_iso);

            r += b_has2.popcount() * w.connectivity_bonus;
            r += b_iso.popcount() * w.isolated_penalty;
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

                const __m128i all_colored = _mm_andnot_si128(oj.m128, board.getOccupied().m128);
                const BitBoard buried_bb(_mm_and_si128(all_colored, s_reg));
                r += buried_bb.popcount() * w.buried_penalty;
            }
        }

        // --- Potential chain score (shared implementation in potential_score.hpp) ---
        if constexpr (CalculatePotential) {
            r += computeMaxPotentialScore(board, packed_heights) * w.potential_score_scale;
        }

        return r;
    }
};

} // namespace puyotan::search