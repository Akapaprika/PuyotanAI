#pragma once

#include <puyotan/core/board.hpp>

namespace puyotan {

class Gravity {
  public:
    static uint32_t execute(Board& board) noexcept;

    /// ★ ヘッダーインライン化: わずか 5命令の SIMD 判定を直接埋め込み
    [[nodiscard]] static __forceinline bool canFall(const Board& board) noexcept {
        const __m128i occ = board.getOccupied().m128;
        const __m128i shifted = _mm_srli_epi64(occ, 1);
        const __m128i boundary = _mm_set1_epi64x(0x8000800080008000ULL);

        const __m128i can_fall_bits =
            _mm_andnot_si128(occ, _mm_andnot_si128(boundary, shifted));

        return !_mm_testz_si128(can_fall_bits, can_fall_bits);
    }
};

} // namespace puyotan