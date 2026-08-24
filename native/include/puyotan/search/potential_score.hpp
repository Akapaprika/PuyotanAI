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

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        // ★ 盤面に 2個以下なら、どこに 1個落としても 4連結発火しないため色ごと即スキップ
        if (bb.popcount() < 3)
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // 隣接マス集合（上下左右）
        const BitBoard neighbor =
            bb.shiftUpRaw() | bb.shiftDownRaw() |
            bb.shiftLeftRaw() | bb.shiftRightRaw();

        const uint64_t n_lo = static_cast<uint64_t>(_mm_cvtsi128_si64(neighbor.m128));
        const uint64_t n_hi = static_cast<uint64_t>(_mm_extract_epi64(neighbor.m128, 1));

        for (int x = 0; x < config::Board::kWidth; ++x) {
            const int h = (packed_heights >> (x << 2)) & 0xFu;
            if (h >= config::Board::kChainableRows)
                continue;

            // ★ スタック書き戻しを完全排除し、レジスタ内ビットテストで一撃判定
            const uint64_t n_lane = (x < 4) ? (n_lo >> (x << 4)) : (n_hi >> ((x - 4) << 4));
            if (((n_lane >> h) & 1ULL) == 0)
                continue;

            // ★ 定数テーブルから 1 命令で直接ロード
            const __m128i point_mask = kPointMasks[x][h].m128;
            const __m128i probed_bb = _mm_or_si128(chainable_bb, point_mask);

            // ★ core と同一の __m128i 直接レジスタ演算
            const __m128i U = _mm_slli_epi64(probed_bb, 1);
            const __m128i D = _mm_srli_epi64(probed_bb, 1);
            const __m128i L = _mm_srli_si128(probed_bb, 2);
            const __m128i R = _mm_slli_si128(probed_bb, 2);

            const __m128i UD_and = _mm_and_si128(U, D);
            const __m128i LR_and = _mm_and_si128(L, R);
            const __m128i UD_or  = _mm_or_si128(U, D);
            const __m128i LR_or  = _mm_or_si128(L, R);

            const __m128i X = _mm_or_si128(UD_and, LR_and);
            const __m128i Y = _mm_and_si128(UD_or, LR_or);

            const __m128i deg_ge3 = _mm_and_si128(probed_bb, _mm_and_si128(X, Y));
            const __m128i deg_ge2 = _mm_and_si128(probed_bb, _mm_or_si128(X, Y));

            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i d2_adj = _mm_and_si128(deg_ge2, _mm_or_si128(u_d2, l_d2));

            const __m128i seeds = _mm_or_si128(deg_ge3, d2_adj);
            if (_mm_testz_si128(seeds, seeds))
                continue;

            // ★ 盤面コピー＆点マスクを SIMD OR 2発で直注入
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