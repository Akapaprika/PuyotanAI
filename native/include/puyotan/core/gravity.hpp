#pragma once

#include <puyotan/core/board.hpp>

namespace puyotan {

class Gravity {
  public:
    static uint32_t execute(Board& board) noexcept;

    /// ★ epi16 を使用することで境界マスク処理を完全消滅
    /// vpsrlw + vptest の実質 2命令（レイテンシ 2サイクル）で完結
    [[nodiscard]] static __forceinline bool canFall(const Board& board) noexcept {
        const __m128i occ = board.getOccupied().m128;
        
        // 16bit単位でシフト。最上位ビット（15bit目）には自動で 0 が入るため境界破壊が起きない
        const __m128i shifted = _mm_srli_epi16(occ, 1);

        // (~occ & shifted) != 0 のとき CF=0（落下可能）
        return _mm_testc_si128(occ, shifted) == 0;
    }
};

} // namespace puyotan