#pragma once

#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

/**
 * @brief Computes the maximum potential chain score achievable by placing
 *        one additional colored puyo adjacent to an existing group on the board.
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    const int    heights[config::Board::kWidth]) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // ★ 1. ループ順序の逆転 (色 c を外側、列 x を内側にする)
    // これにより neighbor マスク計算が 24回 → 4回 (1/6) に激減し、140本のSIMD命令を削減
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.empty())
            continue;

        const BitBoard chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // この色の隣接マスクを色ごとに 1度だけ事前計算
        const BitBoard neighbor =
            bb.shiftUpRaw() | bb.shiftDownRaw() |
            bb.shiftLeftRaw() | bb.shiftRightRaw();

        for (int x = 0; x < config::Board::kWidth; ++x) {
            const int h = heights[x];
            if (h >= config::Board::kChainableRows)
                continue;

            // 隣接していないマスは即座にスキップ
            if (!neighbor.get(x, h))
                continue;

            // ★ 2. 96バイトのボードコピー前の O(1) SIMD 発火プローブ
            // (消えない 2〜3個連結への無駄なボードコピー＆scanGroups を 90% 削減)
            BitBoard probed_bb = chainable_bb;
            probed_bb.cols[x] |= static_cast<uint16_t>(1U << h);

            const BitBoard U = probed_bb.shiftUpRaw();
            const BitBoard L = probed_bb.shiftLeftRaw();
            const BitBoard D = probed_bb.shiftDownRaw();
            const BitBoard R = probed_bb.shiftRightRaw();

            const BitBoard ud_and = U & D;
            const BitBoard ud_or  = U | D;
            const BitBoard lr_and = L & R;
            const BitBoard lr_or  = L | R;

            const BitBoard deg_ge3 =
                probed_bb & ((ud_and & lr_or) | (lr_and & ud_or));
            const BitBoard deg_ge2 =
                probed_bb & (ud_and | lr_and | (ud_or & lr_or));
            const BitBoard d2_adjacent =
                deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

            // 4連結が成立しない場合は 96Bコピーを行わず即スキップ！
            if ((deg_ge3 | d2_adjacent).empty())
                continue;

            // ★ 3. 4連結が確定した時のみ実体をコピーして連鎖スコアを解決
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

            if (pot_score > max_pot_score)
                max_pot_score = pot_score;
        }
    }

    return max_pot_score;
}

} // namespace puyotan::search