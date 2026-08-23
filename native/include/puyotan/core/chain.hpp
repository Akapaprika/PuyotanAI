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

            // 1. 2連結チェック (U | L) と cb の積を PTEST で直接判定 (AND命令を削減)
            const __m128i U = _mm_slli_epi64(cb, 1);
            const __m128i L = _mm_srli_si128(cb, 2);
            if (_mm_testz_si128(cb, _mm_or_si128(U, L))) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const __m128i D = _mm_srli_epi64(cb, 1);
            const __m128i R = _mm_slli_si128(cb, 2);

            // 2. ブール束因数分解による次数判定
            const __m128i X = _mm_or_si128(_mm_and_si128(U, D), _mm_and_si128(L, R));
            const __m128i Y = _mm_and_si128(_mm_or_si128(U, D), _mm_or_si128(L, R));

            const __m128i deg_ge3 = _mm_and_si128(cb, _mm_and_si128(X, Y));
            const __m128i deg_ge2 = _mm_and_si128(cb, _mm_or_si128(X, Y));

            // 3. 対称シード (Symmetric Seeds) の抽出
            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i d_d2 = _mm_srli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i r_d2 = _mm_slli_si128(deg_ge2, 2);
            const __m128i adj_d2 = _mm_or_si128(_mm_or_si128(u_d2, d_d2), _mm_or_si128(l_d2, r_d2));

            const __m128i seeds = _mm_or_si128(deg_ge3, _mm_and_si128(deg_ge2, adj_d2));

            if (_mm_testz_si128(seeds, seeds)) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            // 4. 【ループ完全撤廃】1ステップ Dilation (4連結グループ全体を一括抽出)
            const __m128i u_s = _mm_slli_epi64(seeds, 1);
            const __m128i d_s = _mm_srli_epi64(seeds, 1);
            const __m128i l_s = _mm_srli_si128(seeds, 2);
            const __m128i r_s = _mm_slli_si128(seeds, 2);
            const __m128i adj_s = _mm_or_si128(_mm_or_si128(u_s, d_s), _mm_or_si128(l_s, r_s));

            BitBoard group;
            group.m128 = _mm_and_si128(_mm_or_si128(seeds, adj_s), cb);

            const int sz = group.popcount();

            erasure_data.total_erased.m128 =
                _mm_or_si128(erasure_data.total_erased.m128, group.m128);
            erasure_data.num_erased   += static_cast<uint8_t>(sz);
            erased_color_bits         |= (1u << i);

            // 5. ボーナス加算 (sz < 8 は単一連結成分確定)
            if (__builtin_expect(sz < 8, 1)) {
                erasure_data.group_bonus += kGroupBonusLut[sz];
            } else {
                // 同色で別々の4個グループが同時消去されるレアケースのみ個別分離
                BitBoard rem = group;
                while (!rem.empty()) {
                    BitBoard g = rem.extractLSB();
                    BitBoard p;
                    do {
                        p = g;
                        const __m128i v = g.m128;
                        const __m128i ud = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                        const __m128i lr = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                        g.m128 = _mm_and_si128(_mm_or_si128(v, _mm_or_si128(ud, lr)), group.m128);
                    } while (g != p);

                    const int single_sz = g.popcount();
                    erasure_data.group_bonus += kGroupBonusLut[std::min(single_sz, 15)];
                    rem.andNot(g);
                }
            }

            temp_mask &= (temp_mask - 1);
        }

        erasure_data.num_colors =
            static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

        // 6. おじゃまぷよ消去 (boundary_mask の不要なANDを削除)
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

                // ojama 自体が盤面外ビットを持たないため boundary_mask は不要
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

            const __m128i X = _mm_or_si128(_mm_and_si128(U, D), _mm_and_si128(L, R));
            const __m128i Y = _mm_and_si128(_mm_or_si128(U, D), _mm_or_si128(L, R));

            const __m128i deg_ge3 = _mm_and_si128(cb, _mm_and_si128(X, Y));
            const __m128i deg_ge2 = _mm_and_si128(cb, _mm_or_si128(X, Y));

            // canFire は非対称（U/L のみ）で必要十分
            const __m128i u_d2 = _mm_slli_epi64(deg_ge2, 1);
            const __m128i l_d2 = _mm_srli_si128(deg_ge2, 2);
            const __m128i d2_adj = _mm_and_si128(deg_ge2, _mm_or_si128(u_d2, l_d2));

            const __m128i seeds = _mm_or_si128(deg_ge3, d2_adj);

            if (!_mm_testz_si128(seeds, seeds)) {
                return true;
            }

            temp_mask &= (temp_mask - 1);
        }
        return false;
    }
};

} // namespace puyotan