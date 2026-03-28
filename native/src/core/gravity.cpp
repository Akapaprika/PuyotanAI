#include <puyotan/core/gravity.hpp>
#include <algorithm>

namespace puyotan {

// Mask covering bits 0 through kTotalRows-1 (one column lane: 15 bits = rows 0-14)
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

// ---------------------------------------------------------------------------
// High-Speed Software PEXT (SWAR)
// On AMD Zen 1/1+/2 CPUs (e.g. 3020e), the hardware _pext_u32 is microcoded 
// and takes >250 cycles. This fallback uses BLSI/BLSR equivalents to extract 
// bits in ~10 cycles per column, completely bypassing the hardware latency.
// ---------------------------------------------------------------------------
/**
 * @brief High-Speed Software PEXT (SWAR fallback).
 * 
 * Efficiently extracts bits from 'val' based on 'mask'.
 * On AMD Zen 1/2 CPUs (e.g. 3020e), the hardware _pext_u32 is microcoded 
 * and takes >250 cycles. This fallback uses BLSI/BLSR equivalents to extract 
 * bits in ~10 cycles per column.
 * 
 * @param val Value to extract from.
 * @param mask Mask defining which bits to extract.
 * @return Compacted bits.
 */
static __forceinline uint32_t pext_u16_swar(uint32_t val, uint32_t mask) noexcept {
    uint32_t res = 0;
    int shift = 0;
    while (mask) {
        // Isolate lowest set bit (BLSI equivalent)
        const uint32_t lowest = mask & (0u - mask);
        
        // Branchless check: if val matches the lowest mask bit, shift a 1 into res
        res |= ((val & lowest) != 0) << shift;
        
        ++shift;
        // Clear lowest set bit (BLSR equivalent)
        mask &= (mask - 1);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Per-column PEXT compaction helper.
// UseHi: compile-time flag selecting boards[i].hi (true) or boards[i].lo (false).
//        This replaces the reinterpret_cast+stride pointer arithmetic,
//        giving the compiler full aliasing information at zero runtime cost.
// ---------------------------------------------------------------------------
/**
 * @brief Helper to compact puyos in columns and update all color planes.
 * 
 * @tparam NUM_COLS Number of columns to process (4 for Lo, 2 for Hi).
 * @tparam UseHi Boolean selecting whether to access .hi or .lo segments.
 * @param occ_word Pointer to the occupancy segment.
 * @param boards Pointer to the array of per-color BitBoards.
 * @return Bitmask of colors that fell during compaction.
 */
template<int NUM_COLS, bool UseHi>
static __forceinline uint32_t compactCols(
    uint64_t* __restrict occ_word,
    BitBoard* __restrict boards
) noexcept {
    uint32_t fallen_mask = 0;
    for (int local = 0; local < NUM_COLS; ++local) {
        const int shift = local * config::Board::kBitsPerCol;

        const uint32_t occ_lane = static_cast<uint32_t>(*occ_word >> shift) & kColLaneMask;
        if (occ_lane == 0) continue;

        const int cnt = _mm_popcnt_u32(occ_lane);
        const uint32_t full_occ = (1u << cnt) - 1u;
        const uint32_t new_occ  = full_occ & config::Board::kVisibleColMask;

        if (occ_lane == new_occ) continue; // CRITICAL EARLY OUT: skips entire color loop for this column

        const uint64_t clear = ~(static_cast<uint64_t>(kColLaneMask) << shift);

        for (int i = 0; i < config::Board::kNumColors; ++i) {
            uint64_t& cw = UseHi ? boards[i].hi : boards[i].lo;
            const uint32_t lane = static_cast<uint32_t>(cw >> shift) & kColLaneMask;
            if (lane == 0) continue;

            // Bypass the microcoded BMI2 _pext_u32 on Zen architectures
            const uint32_t compacted = pext_u16_swar(lane, occ_lane) & new_occ;
            fallen_mask |= (compacted != lane) << i;
            cw = (cw & clear) | (static_cast<uint64_t>(compacted) << shift);
        }

        *occ_word = (*occ_word & clear) | (static_cast<uint64_t>(new_occ) << shift);
    }
    return fallen_mask;
}

uint32_t Gravity::execute(Board& board) noexcept {
    const uint32_t m1 = compactCols<config::Board::kColsInLo, false>(
        &board.occupancy_.lo,
        board.boards_.data()
    );

    const uint32_t m2 = compactCols<config::Board::kColsInHi, true>(
        &board.occupancy_.hi,
        board.boards_.data()
    );

    return m1 | m2;
}

bool Gravity::canFall(const Board& board) noexcept {
    const __m128i occ = board.getOccupied().m128;
    const __m128i shifted = _mm_srli_epi64(occ, 1);
    static const __m128i boundary = _mm_set1_epi64x(0x8000800080008000ULL);

    // shifted & ~boundary & ~occ
    const __m128i can_fall_bits = _mm_andnot_si128(occ, _mm_andnot_si128(boundary, shifted));

    // _mm_testz_si128 returns 1 if all bits are 0. We want true if ANY bit is 1.
    return !_mm_testz_si128(can_fall_bits, can_fall_bits);
}

} // namespace puyotan
