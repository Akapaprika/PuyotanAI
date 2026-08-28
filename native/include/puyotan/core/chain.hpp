#pragma once

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
  private:
    // ★ 最適化2: GPR転送と分岐を排除する、完全SIMD化された128bit LSB抽出
    [[nodiscard]] static __forceinline __m128i extractLsbSimd(const __m128i& v) noexcept {
        const __m128i neg = _mm_sub_epi64(_mm_setzero_si128(), v);
        const __m128i lsb64 = _mm_and_si128(v, neg); // [ hi_lsb | lo_lsb ]
        
        const __m128i lo_mask = _mm_set_epi64x(0, -1LL);
        const __m128i lo_only = _mm_and_si128(lsb64, lo_mask); // [ 0 | lo_lsb ]
        
        // lo_lsb が 0 なら全ビット1、非0なら0 のマスクを生成し、上位64bitにコピー
        const __m128i cmp = _mm_cmpeq_epi64(lo_only, _mm_setzero_si128());
        const __m128i hi_mask = _mm_shuffle_epi32(cmp, _MM_SHUFFLE(1, 0, 1, 0));
        
        const __m128i hi_only = _mm_andnot_si128(lo_mask, lsb64); // [ hi_lsb | 0 ]
        return _mm_or_si128(lo_only, _mm_and_si128(hi_only, hi_mask));
    }

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

        // ★ 最適化1: while と TZCNT を排除し、アンロールでアウトオブオーダ実行を最大化
        #pragma unroll
        for (int i = 0; i < config::Rule::kColors; ++i) {
            if ((color_mask & (1u << i)) == 0) continue;

            const Cell c = static_cast<Cell>(i);
            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            const __m128i UL = _mm_or_si128(U, L);
            if (_mm_testz_si128(cb, UL)) continue;

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
            const __m128i ul_d2 = _mm_or_si128(u_d2, l_d2);
            const __m128i ptest_term = _mm_or_si128(XY_and, ul_d2);

            if (_mm_testz_si128(deg_ge2, ptest_term)) continue;

            const __m128i d_d2 = _mm_srli_epi64(deg_ge2, 1);
            const __m128i r_d2 = _mm_slli_si128(deg_ge2, 2);
            const __m128i dr_d2 = _mm_or_si128(d_d2, r_d2);

            const __m128i sym_seeds = _mm_and_si128(deg_ge2, _mm_or_si128(ptest_term, dr_d2));

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

            if (__builtin_expect(sz < 8, 1)) {
                erasure_data.group_bonus += kGroupBonusLut[sz];
            } else {
                // ★ 最適化2適用: 分岐とGPR転送を排除した O(1) LSB抽出
                __m128i g1 = extractLsbSimd(sym_seeds);
                const __m128i grp = group.m128;

                while (true) {
                    const __m128i ud = _mm_or_si128(_mm_slli_epi64(g1, 1), _mm_srli_epi64(g1, 1));
                    const __m128i lr = _mm_or_si128(_mm_slli_si128(g1, 2), _mm_srli_si128(g1, 2));
                    const __m128i next = _mm_and_si128(_mm_or_si128(g1, _mm_or_si128(ud, lr)), grp);
                    if (_mm_testc_si128(g1, next)) break;
                    g1 = next;
                }

                BitBoard b_g1;
                b_g1.m128 = g1;
                const int sz1 = b_g1.popcount();

                if (sz1 == sz) {
                    erasure_data.group_bonus += kGroupBonusLut[std::min(sz, 15)];
                } else {
                    const int sz2 = sz - sz1;
                    erasure_data.group_bonus += kGroupBonusLut[std::min(sz1, 15)];

                    if (sz2 < 8) {
                        erasure_data.group_bonus += kGroupBonusLut[sz2];
                    } else {
                        BitBoard rem = group;
                        rem.andNot(b_g1);
                        while (!rem.empty()) {
                            // ★ 最適化2適用: 残りグループのLSB抽出も完全SIMD化
                            __m128i v_rem = extractLsbSimd(rem.m128);
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
        }

        erasure_data.num_colors =
            static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

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
                    _mm_or_si128(t, oj_erased); // t を再利用して依存関係を短縮
            }
        }
    }

    static __forceinline void applyErasure(Board& board, const ErasureData& data) noexcept {
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            board.boards_[i].andNot(data.total_erased);
        }
        board.occupancy_.andNot(data.total_erased);
    }

    [[nodiscard]] static __forceinline bool canFire(const Board& board,
                                                uint32_t color_mask = kAllColorsMask) noexcept {
        const __m128i chainable_mask = _mm_set_epi64x(
            config::Board::kChainableHiMask, config::Board::kChainableLoMask);

        // ★ 最適化1適用: canFire もアンロールし、各色の発火判定を並行実行させる
        #pragma unroll
        for (int i = 0; i < config::Rule::kColors; ++i) {
            if ((color_mask & (1u << i)) == 0) continue;
            
            const Cell c = static_cast<Cell>(i);
            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            if (_mm_testz_si128(cb, _mm_or_si128(U, L))) continue;

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
            const __m128i ul_d2 = _mm_or_si128(u_d2, l_d2);

            if (!_mm_testz_si128(deg_ge2, _mm_or_si128(XY_and, ul_d2))) {
                return true;
            }
        }
        return false;
    }
};

} // namespace puyotan