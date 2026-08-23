#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/core/board.hpp>

namespace puyotan {

struct alignas(16) ErasureData {
    BitBoard total_erased;
    uint8_t num_erased = 0;
    uint8_t num_colors = 0;
    uint8_t group_bonus = 0;

    __forceinline void clear() noexcept {
        total_erased.m128 = _mm_setzero_si128();
        *reinterpret_cast<uint32_t*>(&num_erased) = 0;
    }
};

class Chain {
  public:
    static constexpr uint32_t kAllColorsMask =
        (1u << config::Rule::kColors) - 1u;

    static constexpr uint8_t kGroupBonusLut[16] = {
        0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10, 10, 10, 10, 10
    };

    static ErasureData execute(Board& board,
                               uint32_t color_mask = kAllColorsMask) noexcept;

    static void execute(Board& board, ErasureData& data,
                        uint32_t color_mask = kAllColorsMask) noexcept;

    static __forceinline void scanGroups(const Board& board, ErasureData& erasure_data,
                                        uint32_t color_mask = kAllColorsMask) noexcept {
        erasure_data.clear();

        const __m128i chainable_mask = _mm_set_epi64x(
            config::Board::kChainableHiMask, config::Board::kChainableLoMask);

        uint32_t erased_color_bits = 0;
        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);

        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            // 1. 2連結チェック (最小の2シフトで即脱出判定)
            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            const __m128i UL = _mm_or_si128(U, L);
            if (_mm_testz_si128(cb, UL)) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const __m128i D = _mm_srli_epi64(cb, 1);
            const __m128i R = _mm_slli_si128(cb, 2);

            // 2. ブール束因数分解 (Zen 1 の 4-wide ベクトル ALU で並列実行)
            const __m128i UD_and = _mm_and_si128(U, D);
            const __m128i LR_and = _mm_and_si128(L, R);
            const __m128i UD_or  = _mm_or_si128(U, D);
            const __m128i LR_or  = _mm_or_si128(L, R);

            const __m128i X = _mm_or_si128(UD_and, LR_and);
            const __m128i Y = _mm_and_si128(UD_or, LR_or);

            const __m128i XY_and  = _mm_and_si128(X, Y);
            const __m128i XY_or   = _mm_or_si128(X, Y);

            const __m128i deg_ge3 = _mm_and_si128(cb, XY_and);
            const __m128i deg_ge2 = _mm_and_si128(cb, XY_or);

            // 3. 【ステージ1: 最速非対称シード判定】(非消去パスは 2シフトのみで即座に脱出)
            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i ul_d2 = _mm_or_si128(u_d2, l_d2);
            const __m128i d2_asym = _mm_and_si128(deg_ge2, ul_d2);

            const __m128i asym_seeds = _mm_or_si128(deg_ge3, d2_asym);
            if (_mm_testz_si128(asym_seeds, asym_seeds)) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            // 4. 【ステージ2: 消去確定時のみ対称シード化＆1ステップ展開】
            const __m128i d_d2 = _mm_srli_epi64(deg_ge2, 1);
            const __m128i r_d2 = _mm_slli_si128(deg_ge2, 2);
            const __m128i dr_d2 = _mm_or_si128(d_d2, r_d2);
            const __m128i adj_d2 = _mm_or_si128(ul_d2, dr_d2);

            const __m128i sym_seeds = _mm_or_si128(deg_ge3, _mm_and_si128(deg_ge2, adj_d2));

            const __m128i u_s = _mm_slli_epi64(sym_seeds, 1);
            const __m128i d_s = _mm_srli_epi64(sym_seeds, 1);
            const __m128i l_s = _mm_srli_si128(sym_seeds, 2);
            const __m128i r_s = _mm_slli_si128(sym_seeds, 2);
            const __m128i adj_s = _mm_or_si128(_mm_or_si128(u_s, d_s), _mm_or_si128(l_s, r_s));

            BitBoard group;
            group.m128 = _mm_and_si128(_mm_or_si128(sym_seeds, adj_s), cb);

            const int sz = group.popcount();

            erasure_data.total_erased.m128 =
                _mm_or_si128(erasure_data.total_erased.m128, group.m128);
            erasure_data.num_erased   += static_cast<uint8_t>(sz);
            erased_color_bits         |= (1u << i);

            // 5. ボーナス加算 (ルール完全厳密・8連結大消しと同色同時消しを正確に分離)
            if (__builtin_expect(sz < 8, 1)) {
                erasure_data.group_bonus += kGroupBonusLut[sz];
            } else {
                // シードから1点を取得
                const uint64_t lo = _mm_cvtsi128_si64(sym_seeds);
                __m128i s0;
                if (lo != 0) {
                    s0 = _mm_cvtsi64_si128(static_cast<int64_t>(lo & -lo));
                } else {
                    const uint64_t hi = _mm_extract_epi64(sym_seeds, 1);
                    s0 = _mm_set_epi64x(static_cast<int64_t>(hi & -hi), 0);
                }

                // 第1グループを完全回収 (PTEST キャリーフラグ直結で最速終了判定)
                const __m128i grp = group.m128;
                __m128i g1 = s0;
                while (true) {
                    const __m128i ud = _mm_or_si128(_mm_slli_epi64(g1, 1), _mm_srli_epi64(g1, 1));
                    const __m128i lr = _mm_or_si128(_mm_slli_si128(g1, 2), _mm_srli_si128(g1, 2));
                    const __m128i next = _mm_and_si128(_mm_or_si128(g1, _mm_or_si128(ud, lr)), grp);
                    if (_mm_testc_si128(g1, next)) break; // 変化がなくなれば終了
                    g1 = next;
                }

                BitBoard b_g1;
                b_g1.m128 = g1;
                const int sz1 = b_g1.popcount();

                if (sz1 == sz) {
                    // ★ 単一の 8連結・10連結等の大消し：100% 正確に満額ボーナスを加算！
                    erasure_data.group_bonus += kGroupBonusLut[std::min(sz, 15)];
                } else {
                    // ★ 同色2ダブ (4+4, 4+5, 5+5 等)：第1グループと第2グループのボーナスを合算
                    const int sz2 = sz - sz1;
                    erasure_data.group_bonus += kGroupBonusLut[std::min(sz1, 15)];

                    if (sz2 < 8) {
                        // 第2グループは単一確定のため 2回目の Flood Fill なしで即加算
                        erasure_data.group_bonus += kGroupBonusLut[sz2];
                    } else {
                        // 3グループ以上の極めて稀なケース (4+4+4 等)
                        BitBoard rem = group;
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
                            BitBoard b_rem;
                            b_rem.m128 = v_rem;
                            const int single_sz = b_rem.popcount();
                            erasure_data.group_bonus += kGroupBonusLut[std::min(single_sz, 15)];
                            rem.andNot(b_rem);
                        }
                    }
                }
            }

            temp_mask &= (temp_mask - 1);
        }

        erasure_data.num_colors =
            static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

        // 6. おじゃまぷよ消去
        if (erasure_data.num_erased > 0) {
            const BitBoard ojama = board.getBitboard(Cell::Ojama);
            if (!ojama.empty()) {
                const __m128i t = erasure_data.total_erased.m128;

                const __m128i raw_up    = _mm_slli_epi64(t, 1);
                const __m128i raw_down  = _mm_srli_epi64(t, 1);
                const __m128i raw_right = _mm_slli_si128(t, 2);
                const __m128i raw_left  = _mm_srli_si128(t, 2);

                const __m128i combined = _mm_or_si128(
                    _mm_or_si128(raw_up, raw_down),
                    _mm_or_si128(raw_right, raw_left));

                const __m128i oj_erased = _mm_and_si128(ojama.m128, combined);
                erasure_data.total_erased.m128 =
                    _mm_or_si128(erasure_data.total_erased.m128, oj_erased);
            }
        }
    }

    static void applyErasure(Board& board, const ErasureData& data) noexcept;

    [[nodiscard]] static __forceinline bool canFire(const Board& board,
                                                uint32_t color_mask = kAllColorsMask) noexcept {
        const __m128i chainable_mask = _mm_set_epi64x(
            config::Board::kChainableHiMask, config::Board::kChainableLoMask);

        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);
        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            if (_mm_testz_si128(cb, _mm_or_si128(U, L))) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const __m128i D = _mm_srli_epi64(cb, 1);
            const __m128i R = _mm_slli_si128(cb, 2);

            const __m128i UD_and = _mm_and_si128(U, D);
            const __m128i LR_and = _mm_and_si128(L, R);
            const __m128i UD_or  = _mm_or_si128(U, D);
            const __m128i LR_or  = _mm_or_si128(L, R);

            const __m128i X = _mm_or_si128(UD_and, LR_and);
            const __m128i Y = _mm_and_si128(UD_or, LR_or);

            const __m128i XY_and = _mm_and_si128(X, Y);
            const __m128i XY_or  = _mm_or_si128(X, Y);

            const __m128i deg_ge2 = _mm_and_si128(cb, XY_or);

            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i d2_adj = _mm_and_si128(deg_ge2, _mm_or_si128(u_d2, l_d2));

            const __m128i seeds_term = _mm_or_si128(XY_and, d2_adj);

            if (!_mm_testz_si128(cb, seeds_term)) {
                return true;
            }

            temp_mask &= (temp_mask - 1);
        }
        return false;
    }
};

} // namespace puyotan