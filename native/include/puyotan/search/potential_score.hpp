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
 * @brief Zen 1 (AMD 3020e) 特化・極限ポテンシャル計算 (μopキャッシュ完全常駐版)
 *        - `_mm_add_epi16` による 1 サイクル全列着地点生成
 *        - `drops[0] | drops[1]` による超高速カラー早期スキップ
 *        - `BLSI/BLSR` による立っているビットのみの最小イテレーション
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    [[maybe_unused]] uint32_t packed_heights = 0) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // 1. 全 6 列の着地点を一括生成 (1 命令)
    const __m128i drop_points_raw = _mm_add_epi16(board.getOccupied().m128, _mm_set1_epi16(1));
    const __m128i drop_points_mask = _mm_and_si128(drop_points_raw, chainable_mask);

    const BitBoard& ojama_bb = board.getBitboard(Cell::Ojama);
    const bool has_ojama = !ojama_bb.empty();

    // 2. 色ループ (4色)
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));

        if (bb.popcount() < 3)
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // 着地点隣接
        const __m128i adj_c = _mm_or_si128(
            _mm_slli_epi64(chainable_bb, 1),
            _mm_or_si128(_mm_slli_si128(chainable_bb, 2), _mm_srli_si128(chainable_bb, 2))
        );

        const __m128i valid_drops = _mm_and_si128(adj_c, drop_points_mask);

        // GPR に 1 ストアで展開
        alignas(16) uint64_t drops[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(drops), valid_drops);

        // 有効な着地点が 1 つもなければ色ごと即座にスキップ (PTESTより高速)
        if ((drops[0] | drops[1]) == 0)
            continue;

        // -------------------------------------------------------------
        // 【疎イテレーション】ビットが立っている箇所のみ 1 サイクルで抽出
        // -------------------------------------------------------------
        for (int part = 0; part < 2; ++part) {
            uint64_t v = drops[part];

            while (v != 0) {
                const uint64_t lsb = v & -v; // BLSI (1 cycle)
                v &= (v - 1);                // BLSR (1 cycle)

                const __m128i pm = (part == 0)
                    ? _mm_cvtsi64_si128(lsb)
                    : _mm_unpacklo_epi64(_mm_setzero_si128(), _mm_cvtsi64_si128(lsb));

                const __m128i probed_bb = _mm_or_si128(chainable_bb, pm);

                // --- 4連結判定 (論理演算最小化) ---
                const __m128i p_U = _mm_slli_epi64(probed_bb, 1);
                const __m128i p_D = _mm_srli_epi64(probed_bb, 1);
                const __m128i p_L = _mm_srli_si128(probed_bb, 2);
                const __m128i p_R = _mm_slli_si128(probed_bb, 2);

                const __m128i p_X = _mm_or_si128(_mm_and_si128(p_U, p_D), _mm_and_si128(p_L, p_R));
                const __m128i p_Y = _mm_and_si128(_mm_or_si128(p_U, p_D), _mm_or_si128(p_L, p_R));

                const __m128i p_deg_ge2 = _mm_and_si128(probed_bb, _mm_or_si128(p_X, p_Y));
                const __m128i p_deg_ge3 = _mm_and_si128(probed_bb, _mm_and_si128(p_X, p_Y));

                const __m128i p_u_d2 = _mm_slli_epi64(p_deg_ge2, 1);
                const __m128i p_l_d2 = _mm_srli_si128(p_deg_ge2, 2);
                const __m128i p_d2_asym = _mm_and_si128(p_deg_ge2, _mm_or_si128(p_u_d2, p_l_d2));

                const __m128i asym_seeds = _mm_or_si128(p_deg_ge3, p_d2_asym);

                if (_mm_testz_si128(asym_seeds, asym_seeds))
                    continue;

                // --- 第1消去グループの抽出 ---
                const __m128i p_d_d2 = _mm_srli_epi64(p_deg_ge2, 1);
                const __m128i p_r_d2 = _mm_slli_si128(p_deg_ge2, 2);
                const __m128i sym_seeds = _mm_or_si128(asym_seeds, 
                    _mm_and_si128(p_deg_ge2, _mm_or_si128(p_d_d2, p_r_d2)));

                const __m128i ud_s = _mm_or_si128(_mm_slli_epi64(sym_seeds, 1), _mm_srli_epi64(sym_seeds, 1));
                const __m128i lr_s = _mm_or_si128(_mm_slli_si128(sym_seeds, 2), _mm_srli_si128(sym_seeds, 2));
                const __m128i adj_s = _mm_or_si128(ud_s, lr_s);

                BitBoard first_group;
                first_group.m128 = _mm_and_si128(_mm_or_si128(sym_seeds, adj_s), probed_bb);

                const int sz = first_group.popcount();
                ErasureData ed;
                ed.total_erased = first_group;
                ed.num_erased   = static_cast<uint8_t>(sz);
                ed.num_colors   = 1;
                ed.group_bonus  = Chain::kGroupBonusLut[std::min(sz, 15)];

                if (has_ojama) {
                    const __m128i t = ed.total_erased.m128;
                    const __m128i ud = _mm_or_si128(_mm_slli_epi64(t, 1), _mm_srli_epi64(t, 1));
                    const __m128i lr = _mm_or_si128(_mm_slli_si128(t, 2), _mm_srli_si128(t, 2));
                    ed.total_erased.m128 = _mm_or_si128(t, _mm_and_si128(ojama_bb.m128, _mm_or_si128(ud, lr)));
                }

                int pot_chain = 1;
                int pot_score = Scorer::calculateStepScore(ed, pot_chain);

                Board temp = board;
                Chain::applyErasure(temp, ed);

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
    }

    return max_pot_score;
}

} // namespace puyotan::search