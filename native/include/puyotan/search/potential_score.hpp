#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

/**
 * @brief Zen 1 (AMD 3020e) 特化・極限ポテンシャル計算 (STLF完全撲滅・μopキャッシュ完全常駐版)
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    [[maybe_unused]] uint32_t packed_heights = 0) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // 全 6 列の着地点を一括生成
    const __m128i drop_points_mask = _mm_and_si128(
        _mm_add_epi16(board.getOccupied().m128, _mm_set1_epi16(1)),
        chainable_mask
    );

    const BitBoard& ojama_bb = board.getBitboard(Cell::Ojama);
    const bool has_ojama = !ojama_bb.empty();

    // 4色ループ
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.popcount() < 3)
            continue;

        // ★ bb.m128 から直接隣接マスク生成 (epi16 で列境界安全・遅延マスキング)
        const __m128i adj_c = _mm_or_si128(
            _mm_slli_epi16(bb.m128, 1),
            _mm_or_si128(_mm_slli_si128(bb.m128, 2), _mm_srli_si128(bb.m128, 2))
        );

        const __m128i valid_drops = _mm_and_si128(adj_c, drop_points_mask);

        // 発火点ゼロ時は即座に次色へ
        if (_mm_testz_si128(valid_drops, valid_drops))
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // ★ STLF ストール完全撲滅 (スタックを使わずレジスタから直接 GPR へ)
        uint64_t lo = static_cast<uint64_t>(_mm_cvtsi128_si64(valid_drops));
        uint64_t hi = static_cast<uint64_t>(_mm_extract_epi64(valid_drops, 1));

        // ★ 単一 while ループ (μop キャッシュ < 650 bytes 常駐)
        while ((lo | hi) != 0) {
            __m128i pm;
            if (lo != 0) {
                const uint64_t lsb = lo & -lo;
                lo &= (lo - 1);
                pm = _mm_cvtsi64_si128(static_cast<int64_t>(lsb));
            } else {
                const uint64_t lsb = hi & -hi;
                hi &= (hi - 1);
                pm = _mm_set_epi64x(static_cast<int64_t>(lsb), 0);
            }

            const __m128i probed_bb = _mm_or_si128(chainable_bb, pm);

            // --- 4連結判定 ---
            const __m128i p_U = _mm_slli_epi16(probed_bb, 1);
            const __m128i p_D = _mm_srli_epi16(probed_bb, 1);
            const __m128i p_L = _mm_srli_si128(probed_bb, 2);
            const __m128i p_R = _mm_slli_si128(probed_bb, 2);

            const __m128i p_UD_and = _mm_and_si128(p_U, p_D);
            const __m128i p_LR_and = _mm_and_si128(p_L, p_R);
            const __m128i p_X      = _mm_or_si128(p_UD_and, p_LR_and);

            const __m128i p_UD_or  = _mm_or_si128(p_U, p_D);
            const __m128i p_LR_or  = _mm_or_si128(p_L, p_R);
            const __m128i p_Y      = _mm_and_si128(p_UD_or, p_LR_or);

            const __m128i p_deg_ge2 = _mm_and_si128(probed_bb, _mm_or_si128(p_X, p_Y));

            const __m128i ptest_term = _mm_or_si128(
                _mm_and_si128(p_X, p_Y),
                _mm_or_si128(_mm_slli_epi16(p_deg_ge2, 1), _mm_srli_si128(p_deg_ge2, 2))
            );

            if (_mm_testz_si128(p_deg_ge2, ptest_term))
                continue;

            // --- 第1消去グループの抽出 ---
            const __m128i sym_seeds = _mm_and_si128(p_deg_ge2, 
                _mm_or_si128(ptest_term, 
                    _mm_or_si128(_mm_srli_epi16(p_deg_ge2, 1), _mm_slli_si128(p_deg_ge2, 2))));

            const __m128i adj_s = _mm_or_si128(
                _mm_or_si128(_mm_slli_epi16(sym_seeds, 1), _mm_srli_epi16(sym_seeds, 1)),
                _mm_or_si128(_mm_slli_si128(sym_seeds, 2), _mm_srli_si128(sym_seeds, 2))
            );

            BitBoard first_group;
            first_group.m128 = _mm_and_si128(_mm_or_si128(sym_seeds, adj_s), probed_bb);

            const int sz = first_group.popcount();
            ErasureData ed;
            ed.total_erased = first_group;
            ed.num_erased   = static_cast<uint8_t>(sz);
            ed.num_colors   = 1;
            ed.group_bonus  = Chain::kGroupBonusLut[std::min(sz, 15)];

            if (has_ojama) {
                const __m128i t  = ed.total_erased.m128;
                const __m128i ud = _mm_or_si128(_mm_slli_epi16(t, 1), _mm_srli_epi16(t, 1));
                const __m128i lr = _mm_or_si128(_mm_slli_si128(t, 2), _mm_srli_si128(t, 2));
                ed.total_erased.m128 = _mm_or_si128(t, _mm_and_si128(ojama_bb.m128, _mm_or_si128(ud, lr)));
            }

            int pot_chain = 1;
            int pot_score = Scorer::calculateStepScore(ed, pot_chain);

            Board temp = board;
            Chain::applySingleErasure(temp, static_cast<Cell>(c), first_group, ed.total_erased, has_ojama);

            uint32_t fallen = Gravity::execute(temp);
            while (fallen != 0) {
                Chain::scanGroups(temp, ed, fallen);
                if (ed.num_erased == 0)
                    break;

                ++pot_chain;
                pot_score += Scorer::calculateStepScore(ed, pot_chain);
                Chain::applyErasure(temp, ed);

                fallen = Gravity::execute(temp);
            }

            max_pot_score = std::max(max_pot_score, pot_score);
        }
    }

    return max_pot_score;
}

} // namespace puyotan::search