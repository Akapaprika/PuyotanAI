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
    float connectivity_bonus = 0.4f;
    float isolated_penalty = -0.6f;
    float buried_penalty = -1.5f;
    float fire_bias = 1.0f;
};

/**
 * @class SoloBeamEvaluator
 * @brief Stateless board scorer for use inside solo beam search.
 */
class SoloBeamEvaluator {
  public:
    /**
     * @brief Evaluate a board state and return a heuristic score for Solo mode.
     */
    static float evaluate(const Board& board,
                          const SoloBeamEvalWeights& w) noexcept {
        float r = 0.0f;

        // --- Precompute all column heights once ---
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

        // --- Potential chain score ---
        int max_pot_score = 0;
        for (int x = 0; x < config::Board::kWidth; ++x) {
            const int h = heights[x];
            if (h >= config::Board::kChainableRows)
                continue;

            for (int c = 0; c < config::Rule::kColors; ++c) {
                const BitBoard& bb =
                    board.getBitboard(static_cast<Cell>(c));
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
        r += static_cast<float>(max_pot_score) * w.potential_score_scale;

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
                          const VsBeamEvalWeights& w) noexcept {
        float r = 0.0f;

        // --- Precompute all column heights once ---
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

            r += static_cast<float>(conn) * w.connectivity_bonus;
            r += static_cast<float>(iso) * w.isolated_penalty;
        }

        // --- Buried puyo count (colored puyos beneath any ojama shadow) ---
        {
            const BitBoard& oj = board.getBitboard(Cell::Ojama);
            if (!oj.empty()) {
                // In-register SIMD downward shadow smearing (eliminates STLF stalls)
                __m128i s_reg = oj.m128;
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 1));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 2));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 4));
                s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 8));

                // Combine all colored boards using parallel register ANDNOT (Occupied & ~Ojama)
                __m128i all_colored = _mm_andnot_si128(oj.m128, board.getOccupied().m128);

                __m128i buried_reg = _mm_and_si128(all_colored, s_reg);

                // Direct register extraction to avoid any memory-store penalties
                uint64_t b_lo = _mm_cvtsi128_si64(buried_reg);
                uint64_t b_hi = _mm_extract_epi64(buried_reg, 1);
                int buried =
                    static_cast<int>(std::popcount(b_lo) + std::popcount(b_hi));
                r += static_cast<float>(buried) * w.buried_penalty;
            }
        }

        // --- Potential chain score ---
        if constexpr (CalculatePotential) {
            int max_pot_score = 0;
            for (int x = 0; x < config::Board::kWidth; ++x) {
                const int h = heights[x]; // cached
                if (h >= config::Board::kChainableRows)
                    continue;

                for (int c = 0; c < config::Rule::kColors; ++c) {
                    const BitBoard& bb =
                        board.getBitboard(static_cast<Cell>(c));
                    // SIMD neighbor mask: 4 shifts replace 3 conditional bb.get() calls.
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

        return r;
    }
};

} // namespace puyotan::search
