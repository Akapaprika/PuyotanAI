#pragma once

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

namespace detail {
// 6列 x 16行 = 96エントリ (1.5KB, L1Dキャッシュに完全常駐)
constexpr auto makePointMasks() {
    std::array<std::array<BitBoard, 16>, config::Board::kWidth> masks{};
    for (int x = 0; x < config::Board::kWidth; ++x) {
        for (int h = 0; h < 16; ++h) {
            if (x < 4) {
                masks[x][h] = BitBoard(1ULL << ((x << 4) + h), 0);
            } else {
                masks[x][h] = BitBoard(0, 1ULL << (((x - 4) << 4) + h));
            }
        }
    }
    return masks;
}
} // namespace detail

inline constexpr auto kPointMasks = detail::makePointMasks();

[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    const int    heights[config::Board::kWidth]) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        // ★ 盤面に 2個以下なら、どこに 1個落としても 4連結発火しないため色ごと即スキップ
        if (bb.popcount() < 3)
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

            // ★ 動的計算を撤廃し、定数テーブルから 1 命令で直接ロード
            const __m128i point_mask = kPointMasks[x][h].m128;

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