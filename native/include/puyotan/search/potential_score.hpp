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

/**
 * @brief Zen 1 (AMD 3020e) 特化型・極限ポテンシャル得点計算
 *        プレフィルタ・発火判定・第1連鎖グループ抽出を一体化（Fused）
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    uint32_t     packed_heights) noexcept
{
    int max_pot_score = 0;
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    // 1. 各列の高さをアンパック
    const int h0 = static_cast<int>((packed_heights      ) & 0xFu);
    const int h1 = static_cast<int>((packed_heights >>  4) & 0xFu);
    const int h2 = static_cast<int>((packed_heights >>  8) & 0xFu);
    const int h3 = static_cast<int>((packed_heights >> 12) & 0xFu);
    const int h4 = static_cast<int>((packed_heights >> 16) & 0xFu);
    const int h5 = static_cast<int>((packed_heights >> 20) & 0xFu);

    const int heights[6] = { h0, h1, h2, h3, h4, h5 };

    // 2. 有効な着地点（12段未満）のみビットを立てた drop_points_mask を 64bit 整数で瞬時に生成
    uint64_t lo_drop = 0;
    if (h0 < config::Board::kChainableRows) lo_drop |= (1ULL << h0);
    if (h1 < config::Board::kChainableRows) lo_drop |= (1ULL << (16 + h1));
    if (h2 < config::Board::kChainableRows) lo_drop |= (1ULL << (32 + h2));
    if (h3 < config::Board::kChainableRows) lo_drop |= (1ULL << (48 + h3));

    uint64_t hi_drop = 0;
    if (h4 < config::Board::kChainableRows) hi_drop |= (1ULL << h4);
    if (h5 < config::Board::kChainableRows) hi_drop |= (1ULL << (16 + h5));

    if ((lo_drop | hi_drop) == 0) [[unlikely]] {
        return 0; // 全列が12段以上（置く場所がない）
    }

    const __m128i drop_points_mask = _mm_set_epi64x(
        static_cast<int64_t>(hi_drop), static_cast<int64_t>(lo_drop));

    // おじゃまぷよ盤面の事前キャッシュ
    const BitBoard& ojama_bb = board.getBitboard(Cell::Ojama);
    const bool has_ojama = !ojama_bb.empty();

    // 3. 色ループ (4色)
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.popcount() < 3)
            continue;

        const __m128i chainable_bb = _mm_and_si128(bb.m128, chainable_mask);

        // ★【超高速 SIMD プレフィルタ】
        // 着地点から見て「下・左・右」のいずれかに同色ぷよが存在する位置を一括計算
        // 着地点から見て：下=自身の上ビット, 左=右ビット, 右=左ビット
        const __m128i adj_c = _mm_or_si128(
            _mm_slli_epi64(chainable_bb, 1),
            _mm_or_si128(_mm_slli_si128(chainable_bb, 2), _mm_srli_si128(chainable_bb, 2))
        );

        const __m128i valid_drops = _mm_and_si128(adj_c, drop_points_mask);

        // 6列すべてで隣接ぷよが存在しなければ、この色は 1 サイクルで丸ごとスキップ！
        if (_mm_testz_si128(valid_drops, valid_drops))
            continue;

        const uint64_t v_lo = static_cast<uint64_t>(_mm_cvtsi128_si64(valid_drops));
        const uint64_t v_hi = static_cast<uint64_t>(_mm_extract_epi64(valid_drops, 1));

        #pragma unroll 6
        for (int x = 0; x < config::Board::kWidth; ++x) {
            // 列 x の着地点が有効（隣接ぷよあり）かビットテスト（1サイクル未満）
            const bool is_valid = (x < 4)
                ? ((v_lo & (1ULL << ((x << 4) + heights[x]))) != 0)
                : ((v_hi & (1ULL << (((x - 4) << 4) + heights[x]))) != 0);

            if (!is_valid)
                continue;

            const int h = heights[x];
            const __m128i point_mask = kPointMasks[x][h].m128;
            const __m128i probed_bb = _mm_or_si128(chainable_bb, point_mask);

            // --- 厳密な 4連結シード判定（Zen 1 4-wide ベクトル ALU 最適化） ---
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

            const __m128i p_deg_ge2 = _mm_and_si128(probed_bb, p_XY_or);

            const __m128i p_u_d2 = _mm_slli_epi64(p_deg_ge2, 1);
            const __m128i p_l_d2 = _mm_srli_si128(p_deg_ge2, 2);
            const __m128i p_ul_d2 = _mm_or_si128(p_u_d2, p_l_d2);
            const __m128i p_d2_asym = _mm_and_si128(p_deg_ge2, p_ul_d2);

            // ステージ1: 最速非対称シード脱出判定
            if (_mm_testz_si128(probed_bb, _mm_or_si128(p_XY_and, p_d2_asym)))
                continue;

            // ★ ステージ2: 消去確定！シードから第1消去グループをそのまま抽出（二重 scanGroups を完全スキップ）
            const __m128i p_deg_ge3 = _mm_and_si128(probed_bb, p_XY_and);
            const __m128i p_d_d2 = _mm_srli_epi64(p_deg_ge2, 1);
            const __m128i p_r_d2 = _mm_slli_si128(p_deg_ge2, 2);
            const __m128i p_dr_d2 = _mm_or_si128(p_d_d2, p_r_d2);
            const __m128i p_adj_d2 = _mm_or_si128(p_ul_d2, p_dr_d2);

            const __m128i sym_seeds = _mm_or_si128(p_deg_ge3, _mm_and_si128(p_deg_ge2, p_adj_d2));

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

            // 第1連鎖ボーナス計算（ルール完全厳密）
            if (__builtin_expect(sz < 8, 1)) {
                ed.group_bonus = Chain::kGroupBonusLut[sz];
            } else {
                // 8連結大消しと同色複数グループの厳密分離
                const uint64_t lo_s = static_cast<uint64_t>(_mm_cvtsi128_si64(sym_seeds));
                __m128i s0;
                if (lo_s != 0) {
                    s0 = _mm_cvtsi64_si128(static_cast<int64_t>(lo_s & -lo_s));
                } else {
                    const uint64_t hi_s = static_cast<uint64_t>(_mm_extract_epi64(sym_seeds, 1));
                    s0 = _mm_set_epi64x(static_cast<int64_t>(hi_s & -hi_s), 0);
                }

                const __m128i grp = first_group.m128;
                __m128i g1 = s0;
                while (true) {
                    const __m128i ud = _mm_or_si128(_mm_slli_epi64(g1, 1), _mm_srli_epi64(g1, 1));
                    const __m128i lr = _mm_or_si128(_mm_slli_si128(g1, 2), _mm_srli_si128(g1, 2));
                    const __m128i next = _mm_and_si128(_mm_or_si128(g1, _mm_or_si128(ud, lr)), grp);
                    if (_mm_testc_si128(g1, next)) break;
                    g1 = next;
                }

                BitBoard b_g1(g1);
                const int sz1 = b_g1.popcount();
                if (sz1 == sz) {
                    ed.group_bonus = Chain::kGroupBonusLut[std::min(sz, 15)];
                } else {
                    const int sz2 = sz - sz1;
                    ed.group_bonus = Chain::kGroupBonusLut[std::min(sz1, 15)];
                    if (sz2 < 8) {
                        ed.group_bonus += Chain::kGroupBonusLut[sz2];
                    } else {
                        BitBoard rem = first_group;
                        rem.andNot(b_g1);
                        while (!rem.empty()) {
                            BitBoard g_rem = rem.extractLSB();
                            __m128i v_rem = g_rem.m128;
                            while (true) {
                                const __m128i ud = _mm_or_si128(_mm_slli_epi64(v_rem, 1), _mm_srli_epi64(v_rem, 1));
                                const __m128i lr = _mm_or_si128(_mm_slli_si128(v_rem, 2), _mm_srli_si128(v_rem, 2));
                                const __m128i next = _mm_and_si128(_mm_or_si128(v_rem, _mm_or_si128(ud, lr)), grp);
                                if (_mm_testc_si128(v_rem, next)) break;
                                v_rem = next;
                            }
                            BitBoard b_rem(v_rem);
                            ed.group_bonus += Chain::kGroupBonusLut[std::min(b_rem.popcount(), 15)];
                            rem.andNot(b_rem);
                        }
                    }
                }
            }

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

            // 発火確定時のみ盤面を複製して第1連鎖消去を適用
            Board temp = board;
            temp.dropMask(static_cast<Cell>(c), kPointMasks[x][h]);
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