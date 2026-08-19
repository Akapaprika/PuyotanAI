#pragma once

#include <algorithm>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    const int    heights[config::Board::kWidth]) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.empty())
            continue;

        const BitBoard chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        const BitBoard neighbor =
            bb.shiftUpRaw() | bb.shiftDownRaw() |
            bb.shiftLeftRaw() | bb.shiftRightRaw();

        for (int x = 0; x < config::Board::kWidth; ++x) {
            const int h = heights[x];
            if (h >= config::Board::kChainableRows)
                continue;

            if (!neighbor.get(x, h))
                continue;

            // =========================================================
            // ★ 128ビット一括合成 (Store-to-Load Forwarding Stall ゼロ)
            // =========================================================
            const __m128i point_mask = (x < 4)
                ? _mm_set_epi64x(0, 1ULL << ((x << 4) + h))
                : _mm_set_epi64x(1ULL << (((x - 4) << 4) + h), 0);

            const BitBoard probed_bb = _mm_or_si128(chainable_bb.m128, point_mask);

            const BitBoard U = probed_bb.shiftUpRaw();
            const BitBoard L = probed_bb.shiftLeftRaw();
            const BitBoard D = probed_bb.shiftDownRaw();
            const BitBoard R = probed_bb.shiftRightRaw();

            // ★ 因数分解による O(1) 発火プローブ
            const BitBoard X = (U & D) | (L & R);
            const BitBoard Y = (U | D) & (L | R);

            const BitBoard deg_ge3 = probed_bb & (X & Y);
            const BitBoard deg_ge2 = probed_bb & (X | Y);
            const BitBoard d2_adjacent =
                deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

            if ((deg_ge3 | d2_adjacent).empty())
                continue;

            Board temp = board;
            temp.dropNewPiece(x, h, static_cast<Cell>(c));

            ErasureData ed;
            Chain::scanGroups(temp, ed, 1u << c);

            int pot_chain = 0, pot_score = 0;
            while (ed.num_erased > 0) {
                ++pot_chain;
                pot_score += Scorer::calculateStepScore(ed, pot_chain);
                Chain::applyErasure(temp, ed);

                const uint32_t fallen = Gravity::execute(temp);
                if (fallen == 0) [[unlikely]]
                    break;
                Chain::scanGroups(temp, ed, fallen);
            }

            max_pot_score = std::max(max_pot_score, pot_score);
        }
    }

    return max_pot_score;
}

} // namespace puyotan::search