#pragma once

#include <puyotan/core/board.hpp>

namespace puyotan {

class Gravity {
  public:
    static uint32_t execute(Board& board) noexcept;

    /// ★ PTEST の CF フラグ機能により、ANDNOT を 1命令削減（5命令 → 4命令）
    [[nodiscard]] static __forceinline bool canFall(const Board& board) noexcept {
        const __m128i occ = board.getOccupied().m128;
        const __m128i shifted = _mm_srli_epi64(occ, 1);
        const __m128i boundary = _mm_set1_epi64x(0x8000800080008000ULL);
        const __m128i valid_shifted = _mm_andnot_si128(boundary, shifted);

        return _mm_testc_si128(occ, valid_shifted) == 0;
    }
};

} // namespace puyotan