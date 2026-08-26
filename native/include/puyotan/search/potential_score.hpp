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
    uint32_t     packed_heights) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // 1. 高さを色ループの外で事前アンパック
    const int heights[6] = {
        static_cast<int>((packed_heights      ) & 0xFu),
        static_cast<int>((packed_heights >>  4) & 0xFu),
        static_cast<int>((packed_heights >>  8) & 0xFu),
        static_cast<int>((packed_heights >> 12) & 0xFu),
        static_cast<int>((packed_heights >> 16) & 0xFu),
        static_cast<int>((packed_heights >> 20) & 0xFu)
    };

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.popcount() < 3)
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        #pragma unroll 6
        for (int x = 0; x < config::Board::kWidth; ++x) {
            const int h = heights[x];
            if (h >= config::Board::kChainableRows)
                continue;

            const __m128i point_mask = kPointMasks[x][h].m128;
            const __m128i probed_bb = _mm_or_si128(chainable_bb, point_mask);

            // --- 厳密な 4連結シード判定 ---
            const __m128i p_U = _mm_slli_epi64(probed_bb, 1);
            const __m128i p_D = _mm_srli_epi64(probed_bb, 1);
            const __m128i p_L = _mm_srli_si128(probed_bb, 2);
            const __m128i p_R = _mm_slli_si128(probed_bb, 2);

            const __m128i p_UD_and = _mm_and_si128(p_U, p_D);
            const __m128i p_LR_and = _mm_and_si128(p_L, p_R);
            const __m128i p_UD_or  = _mm_or_si128(p_U, p_D);
            const __m128i p_LR_or  = _mm_or_si128(p_L, p_R);

            const __m128i p_X = _mm_or_si128(p_UD_and, p_LR_and);
            const __m128i p_Y = _mm_and_si128(p_UD_or, p_LR_or);

            const __m128i p_deg_ge3 = _mm_and_si128(probed_bb, _mm_and_si128(p_X, p_Y));
            const __m128i p_deg_ge2 = _mm_and_si128(probed_bb, _mm_or_si128(p_X, p_Y));

            const __m128i p_u_d2 = _mm_slli_epi64(p_deg_ge2, 1);
            const __m128i p_l_d2 = _mm_srli_si128(p_deg_ge2, 2);
            const __m128i p_d2_adj = _mm_and_si128(p_deg_ge2, _mm_or_si128(p_u_d2, p_l_d2));

            const __m128i seeds = _mm_or_si128(p_deg_ge3, p_d2_adj);
            if (_mm_testz_si128(seeds, seeds))
                continue;

            // 連鎖シミュレーション（クイック判定付き）
            Board temp = board;
            temp.dropMask(static_cast<Cell>(c), kPointMasks[x][h]);

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