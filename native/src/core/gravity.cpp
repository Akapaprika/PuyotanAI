#include <algorithm>
#include <puyotan/core/gravity.hpp>

namespace puyotan {

static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

/**
 * @brief 演算回数を限界まで削った超高速版 compactCols
 * occ_lane のビット解体（whileループ）を1列あたり1回のみ行い、全5色で使い回すことで演算数を約60%削減。
 */
template <int NUM_COLS, bool UseHi>
static __forceinline uint32_t
compactCols(uint64_t* __restrict occ_word,
            BitBoard* __restrict boards) noexcept {
    uint32_t fallen_mask = 0;

    for (int local = 0; local < NUM_COLS; ++local) {
        const int shift = local * config::Board::kBitsPerCol;

        const uint32_t occ_lane =
            static_cast<uint32_t>(*occ_word >> shift) & kColLaneMask;

        // 隙間がない（落下不要の）列は即座にスキップ（最速パス）
        if (((occ_lane + 1) & occ_lane) == 0)
            continue;

        const int cnt = _mm_popcnt_u32(occ_lane);
        const uint32_t full_occ = (1u << cnt) - 1u;
        const uint32_t new_occ = full_occ & config::Board::kVisibleColMask;
        const uint64_t clear = ~(static_cast<uint64_t>(kColLaneMask) << shift);

        // ★【演算削減①】occ_lane のビット解体（whileループ）を1列あたり「1回だけ」実行
        uint32_t bit_masks[16];
        int num_bits = 0;
        uint32_t m = occ_lane;
        while (m) {
            bit_masks[num_bits++] = m & (0u - m); // 最下位ビットの抽出
            m &= (m - 1);                         // 最下位ビットの消去
        }

        // ★【演算削減②】抽出したビット位置を全5色で使い回す（重複演算を完全排除）
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            uint64_t& cw = UseHi ? boards[i].hi : boards[i].lo;
            const uint32_t lane =
                static_cast<uint32_t>(cw >> shift) & kColLaneMask;
            if (lane == 0)
                continue;

            // 分岐なしの超高速ビット圧縮（whileループなし）
            uint32_t compacted = 0;
            for (int b = 0; b < num_bits; ++b) {
                compacted |= ((lane & bit_masks[b]) != 0) << b;
            }
            compacted &= new_occ;

            fallen_mask |= (compacted != lane) << i;
            cw = (cw & clear) | (static_cast<uint64_t>(compacted) << shift);
        }

        *occ_word =
            (*occ_word & clear) | (static_cast<uint64_t>(new_occ) << shift);
    }
    return fallen_mask;
}

uint32_t Gravity::execute(Board& board) noexcept {
    const uint32_t m1 = compactCols<config::Board::kColsInLo, false>(
        &board.occupancy_.lo, board.boards_.data());

    const uint32_t m2 = compactCols<config::Board::kColsInHi, true>(
        &board.occupancy_.hi, board.boards_.data());

    return m1 | m2;
}

bool Gravity::canFall(const Board& board) noexcept {
    const __m128i occ = board.getOccupied().m128;
    const __m128i shifted = _mm_srli_epi64(occ, 1);
    const __m128i boundary = _mm_set1_epi64x(0x8000800080008000ULL);

    const __m128i can_fall_bits =
        _mm_andnot_si128(occ, _mm_andnot_si128(boundary, shifted));

    return !_mm_testz_si128(can_fall_bits, can_fall_bits);
}

} // namespace puyotan