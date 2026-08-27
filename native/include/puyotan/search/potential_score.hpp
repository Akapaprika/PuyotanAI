#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

namespace detail {
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

/**
 * @brief Zen 1 (AMD 3020e) 特化型・極限ポテンシャル得点計算 (理論極限版)
 *        - 完全無分岐 drop_points 生成
        - グラフ理論的性質による第1連鎖グループ計算の O(1) 化
        - Port 0 / Port 1 完全均等パイプライン最適化
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    uint32_t     packed_heights) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // 1. 各列の高さをアンパック (4bit x 6)
    const int h0 = static_cast<int>((packed_heights      ) & 0xFu);
    const int h1 = static_cast<int>((packed_heights >>  4) & 0xFu);
    const int h2 = static_cast<int>((packed_heights >>  8) & 0xFu);
    const int h3 = static_cast<int>((packed_heights >> 12) & 0xFu);
    const int h4 = static_cast<int>((packed_heights >> 16) & 0xFu);
    const int h5 = static_cast<int>((packed_heights >> 20) & 0xFu);

    const int heights[6] = { h0, h1, h2, h3, h4, h5 };

    // 2. 有効着地点マスクを【完全無分岐】で生成
    //    h >= 12 のビットは 12~15 bit 目に立つため、chainable_mask の AND 1 回で自動消去される
    const uint64_t raw_lo = (1ULL << h0) | (1ULL << (16 + h1)) | 
                            (1ULL << (32 + h2)) | (1ULL << (48 + h3));
    const uint64_t raw_hi = (1ULL << h4) | (1ULL << (16 + h5));

    const __m128i drop_points_mask = _mm_and_si128(
        _mm_set_epi64x(static_cast<int64_t>(raw_hi), static_cast<int64_t>(raw_lo)),
        chainable_mask);

    // おじゃまぷよ盤面の事前キャッシュ
    const BitBoard& ojama_bb = board.getBitboard(Cell::Ojama);
    const bool has_ojama = !ojama_bb.empty();

    // 3. 色ループ (4色)
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.popcount() < 3)
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // 着地点から見た隣接（下・左・右）を一括計算
        const __m128i adj_c = _mm_or_si128(
            _mm_slli_epi64(chainable_bb, 1),
            _mm_or_si128(_mm_slli_si128(chainable_bb, 2), _mm_srli_si128(chainable_bb, 2))
        );

        const __m128i valid_drops = _mm_and_si128(adj_c, drop_points_mask);

        // 6列すべてで隣接がなければ 1 命令でスキップ
        if (_mm_testz_si128(valid_drops, valid_drops))
            continue;

        const uint64_t v_lo = static_cast<uint64_t>(_mm_cvtsi128_si64(valid_drops));
        const uint64_t v_hi = static_cast<uint64_t>(_mm_extract_epi64(valid_drops, 1));

        #pragma unroll 6
        for (int x = 0; x < config::Board::kWidth; ++x) {
            // 列 x の 16bit 領域が 0 でなければ有効（動的ビットシフトを完全排除）
            const bool is_valid = (x < 4)
                ? ((v_lo & (0xFFFFULL << (x << 4))) != 0)
                : ((v_hi & (0xFFFFULL << ((x - 4) << 4))) != 0);

            if (!is_valid)
                continue;

            const int h = heights[x];
            const __m128i point_mask = kPointMasks[x][h].m128;
            const __m128i probed_bb = _mm_or_si128(chainable_bb, point_mask);

            // --- 4連結シード判定（Port 0 / Port 1 最適インターリーブ） ---
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

            const __m128i p_XY_and = _mm_and_si128(p_X, p_Y);
            const __m128i p_XY_or  = _mm_or_si128(p_X, p_Y);

            // --- Stage 1: 4連結シード判定 & 非対称早期脱出 ---
            const __m128i p_deg_ge2 = _mm_and_si128(probed_bb, p_XY_or);
            const __m128i p_deg_ge3 = _mm_and_si128(probed_bb, p_XY_and);

            const __m128i p_u_d2 = _mm_slli_epi64(p_deg_ge2, 1);
            const __m128i p_l_d2 = _mm_srli_si128(p_deg_ge2, 2);
            const __m128i p_ul_d2 = _mm_or_si128(p_u_d2, p_l_d2);
            const __m128i p_d2_asym = _mm_and_si128(p_deg_ge2, p_ul_d2);

            // 分配律により、この asym_seeds を Stage 2 でそのまま再利用する
            const __m128i asym_seeds = _mm_or_si128(p_deg_ge3, p_d2_asym);

            if (_mm_testz_si128(asym_seeds, asym_seeds))
                continue;

            // --- Stage 2: 消去確定！シードから第1消去グループをそのまま抽出 ---
            const __m128i p_d_d2 = _mm_srli_epi64(p_deg_ge2, 1);
            const __m128i p_r_d2 = _mm_slli_si128(p_deg_ge2, 2);
            const __m128i p_dr_d2 = _mm_or_si128(p_d_d2, p_r_d2);

            // 【分配律等式】: sym_seeds = asym_seeds | (p_deg_ge2 & p_dr_d2)
            // (p_deg_ge3 の再計算と p_adj_d2 の OR 結合を完全にスキップ)
            const __m128i sym_seeds = _mm_or_si128(asym_seeds, _mm_and_si128(p_deg_ge2, p_dr_d2));

            const __m128i u_s = _mm_slli_epi64(sym_seeds, 1);
            const __m128i d_s = _mm_srli_epi64(sym_seeds, 1);
            const __m128i l_s = _mm_srli_si128(sym_seeds, 2);
            const __m128i r_s = _mm_slli_si128(sym_seeds, 2);
            const __m128i adj_s = _mm_or_si128(_mm_or_si128(u_s, d_s), _mm_or_si128(l_s, r_s));

            BitBoard first_group;
            first_group.m128 = _mm_and_si128(_mm_or_si128(sym_seeds, adj_s), probed_bb);

            const int sz = first_group.popcount();
            ErasureData ed;
            ed.total_erased = first_group;
            ed.num_erased   = static_cast<uint8_t>(sz);
            ed.num_colors   = 1;
            // 1個置きで発生する4連結グループは数学的に1個しか存在し得ないため O(1) で厳密計算完了
            ed.group_bonus  = Chain::kGroupBonusLut[std::min(sz, 15)];

            // 第1連鎖のおじゃま消去
            if (has_ojama) {
                const __m128i t = ed.total_erased.m128;
                const __m128i raw_up    = _mm_slli_epi64(t, 1);
                const __m128i raw_down  = _mm_srli_epi64(t, 1);
                const __m128i raw_right = _mm_slli_si128(t, 2);
                const __m128i raw_left  = _mm_srli_si128(t, 2);
                const __m128i combined  = _mm_or_si128(_mm_or_si128(raw_up, raw_down), _mm_or_si128(raw_right, raw_left));
                const __m128i oj_erased = _mm_and_si128(ojama_bb.m128, combined);
                ed.total_erased.m128 = _mm_or_si128(ed.total_erased.m128, oj_erased);
            }

            // 第1連鎖スコア加算
            int pot_chain = 1;
            int pot_score = Scorer::calculateStepScore(ed, pot_chain);

            // 発火確定時のみ盤面を複製して消去適用（dropMask は ed.total_erased に含まれるため不要）
            Board temp = board;
            Chain::applyErasure(temp, ed);

            // 重力落下 & 第2連鎖以降のシミュレーションループ
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