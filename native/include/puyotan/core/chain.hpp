#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>

namespace puyotan {

struct alignas(16) ErasureData {
    BitBoard total_erased;
    union {
        struct {
            uint8_t num_erased;
            uint8_t num_colors;
            uint8_t group_bonus;
            uint8_t _pad;
        };
        uint32_t meta_u32 = 0; // 32bit一括アクセス用
    };

    // ★ 1命令（mov dword ptr）で安全に一括ゼロクリア
    __forceinline void clear() noexcept {
        total_erased.m128 = _mm_setzero_si128();
        meta_u32 = 0;
    }
};

class Chain {
  public:
    // ★ 16バイト整列された静的マスク定数（関数の先頭より前で定義）
    alignas(16) static constexpr uint64_t kChainableMask[2] = {
        config::Board::kChainableLoMask,
        config::Board::kChainableHiMask
    };

    static constexpr uint32_t kAllColorsMask =
        (1u << config::Rule::kColors) - 1u;

    static constexpr uint8_t kGroupBonusLut[16] = {
        0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10, 10, 10, 10, 10
    };

    // ★ ヘッダ内完全インライン化（CALL/RET 消滅）
    [[nodiscard]] static __forceinline ErasureData execute(Board& board,
                               uint32_t color_mask = kAllColorsMask) noexcept {
        ErasureData data;
        execute(board, data, color_mask);
        return data;
    }

    static __forceinline void execute(Board& board, ErasureData& data,
                        uint32_t color_mask = kAllColorsMask) noexcept {
        scanGroups(board, data, color_mask);
        if (data.num_erased > 0) {
            applyErasure(board, data);
        }
    }

    static __forceinline void scanGroups(const Board& board, ErasureData& erasure_data,
                                        uint32_t color_mask = kAllColorsMask) noexcept {
        erasure_data.clear();

        // アライメント保証された 128bit ロード
        const __m128i chainable_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(kChainableMask));

        uint32_t erased_color_bits = 0;
        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);

        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            // 1. 全方向シフトを同一クロックで一斉発行（FP0, FP1, FP2 を同時飽和）
            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i D = _mm_srli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            const __m128i R = _mm_slli_si128(cb, 2);

            // 2連結チェック (D, R はすでにレジスタに到着済み)
            if (_mm_testz_si128(cb, _mm_or_si128(U, L))) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            // 2. ブール束因数分解 (FP0/FP1 デュアルイシュー配置)
            const __m128i v_and = _mm_and_si128(U, D);
            const __m128i h_and = _mm_and_si128(L, R);
            const __m128i v_or  = _mm_or_si128(U, D);
            const __m128i h_or  = _mm_or_si128(L, R);

            const __m128i X = _mm_or_si128(v_and, h_and);
            const __m128i Y = _mm_and_si128(v_or, h_or);

            const __m128i XY_and  = _mm_and_si128(X, Y);
            const __m128i deg_ge2 = _mm_and_si128(cb, _mm_or_si128(X, Y));

            // 3. 次数2以上が上または左に隣接しているか
            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i ptest_term = _mm_or_si128(XY_and, _mm_or_si128(u_d2, l_d2));

            if (_mm_testz_si128(deg_ge2, ptest_term)) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            // 4. 対称シードの復元
            const __m128i d_d2 = _mm_srli_epi64(deg_ge2, 1);
            const __m128i r_d2 = _mm_slli_si128(deg_ge2, 2);
            const __m128i sym_seeds = _mm_and_si128(deg_ge2, 
                _mm_or_si128(ptest_term, _mm_or_si128(d_d2, r_d2)));

            // 5. 消去グループ展開
            const __m128i ud_s = _mm_or_si128(_mm_slli_epi64(sym_seeds, 1), _mm_srli_epi64(sym_seeds, 1));
            const __m128i lr_s = _mm_or_si128(_mm_slli_si128(sym_seeds, 2), _mm_srli_si128(sym_seeds, 2));
            const __m128i group_m128 = _mm_and_si128(_mm_or_si128(sym_seeds, _mm_or_si128(ud_s, lr_s)), cb);

            const uint64_t g_lo = _mm_cvtsi128_si64(group_m128);
            const uint64_t g_hi = _mm_extract_epi64(group_m128, 1);
            const int sz = static_cast<int>(_mm_popcnt_u64(g_lo) + _mm_popcnt_u64(g_hi));

            erasure_data.total_erased.m128 =
                _mm_or_si128(erasure_data.total_erased.m128, group_m128);
            erasure_data.num_erased   += static_cast<uint8_t>(sz);
            erased_color_bits         |= (1u << i);

            // 6. ボーナス加算
            if (__builtin_expect(sz < 8, 1)) {
                erasure_data.group_bonus += kGroupBonusLut[sz];
            } else {
                auto flood_fill = [&](__m128i seed, __m128i mask) noexcept -> __m128i {
                    __m128i curr = seed;
                    while (true) {
                        const __m128i ud = _mm_or_si128(_mm_slli_epi64(curr, 1), _mm_srli_epi64(curr, 1));
                        const __m128i lr = _mm_or_si128(_mm_slli_si128(curr, 2), _mm_srli_si128(curr, 2));
                        const __m128i next = _mm_and_si128(_mm_or_si128(curr, _mm_or_si128(ud, lr)), mask);
                        if (_mm_testc_si128(curr, next)) break;
                        curr = next;
                    }
                    return curr;
                };

                const uint64_t lo = _mm_cvtsi128_si64(sym_seeds);
                const uint64_t hi = _mm_extract_epi64(sym_seeds, 1);
                const uint64_t s_lo = lo & (0ULL - lo);
                const uint64_t s_hi = (lo == 0) ? (hi & (0ULL - hi)) : 0ULL;
                const __m128i s0 = _mm_set_epi64x(s_hi, s_lo);
                const __m128i g1 = flood_fill(s0, group_m128);
                BitBoard b_g1;
                b_g1.m128 = g1;
                const int sz1 = b_g1.popcount();

                erasure_data.group_bonus += kGroupBonusLut[std::min(sz1, 15)];

                if (sz1 < sz) {
                    const int sz2 = sz - sz1;
                    if (sz2 < 8) {
                        erasure_data.group_bonus += kGroupBonusLut[sz2];
                    } else {
                        BitBoard rem(group_m128);
                        rem.andNot(b_g1);
                        while (!rem.empty()) {
                            BitBoard g_rem = rem.extractLSB();
                            const __m128i v_rem = flood_fill(g_rem.m128, group_m128);
                            BitBoard b_rem;
                            b_rem.m128 = v_rem;
                            erasure_data.group_bonus += kGroupBonusLut[std::min(b_rem.popcount(), 15)];
                            rem.andNot(b_rem);
                        }
                    }
                }
            }

            temp_mask &= (temp_mask - 1);
        }

        erasure_data.num_colors =
            static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

        // 7. おじゃまぷよ消去
        if (erasure_data.num_erased > 0) {
            const BitBoard ojama = board.getBitboard(Cell::Ojama);
            if (!ojama.empty()) {
                const __m128i t = erasure_data.total_erased.m128;
                const __m128i ud = _mm_or_si128(_mm_slli_epi64(t, 1), _mm_srli_epi64(t, 1));
                const __m128i lr = _mm_or_si128(_mm_slli_si128(t, 2), _mm_srli_si128(t, 2));
                const __m128i oj_erased = _mm_and_si128(ojama.m128, _mm_or_si128(ud, lr));
                erasure_data.total_erased.m128 = _mm_or_si128(t, oj_erased);
            }
        }
    }

    // ★ 完全ストレートライン（分岐ゼロ・6サイクル決定論的ストア）
    static __forceinline void applyErasure(Board& board, const ErasureData& data) noexcept {
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            board.boards_[i].andNot(data.total_erased);
        }
        board.occupancy_.andNot(data.total_erased);
    }

    [[nodiscard]] static __forceinline bool canFire(const Board& board,
                                                uint32_t color_mask = kAllColorsMask) noexcept {
        const __m128i chainable_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(kChainableMask));

        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);
        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const __m128i cb = _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i D = _mm_srli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            const __m128i R = _mm_slli_si128(cb, 2);

            if (_mm_testz_si128(cb, _mm_or_si128(U, L))) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const __m128i X = _mm_or_si128(_mm_and_si128(U, D), _mm_and_si128(L, R));
            const __m128i Y = _mm_and_si128(_mm_or_si128(U, D), _mm_or_si128(L, R));

            const __m128i deg_ge2 = _mm_and_si128(cb, _mm_or_si128(X, Y));
            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);

            if (!_mm_testz_si128(deg_ge2, _mm_or_si128(_mm_and_si128(X, Y), _mm_or_si128(u_d2, l_d2)))) {
                return true;
            }

            temp_mask &= (temp_mask - 1);
        }
        return false;
    }

    static __forceinline void applySingleErasure(Board& board, Cell color, 
        const BitBoard& group, 
        const BitBoard& total_erased, 
        bool has_ojama) noexcept {
        board.boards_[static_cast<int>(color)].andNot(group);
        if (has_ojama) {
            board.boards_[static_cast<int>(Cell::Ojama)].andNot(total_erased);
        }
        board.occupancy_.andNot(total_erased);
    }
};

} // namespace puyotan