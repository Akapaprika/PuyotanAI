#include <puyotan/core/gravity.hpp>
#include <algorithm>

namespace puyotan {

// Mask covering bits 0 through kTotalRows-1 (one column lane: 15 bits = rows 0-14)
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

// ---------------------------------------------------------------------------
// Per-column PEXT compaction helper.
// ---------------------------------------------------------------------------
template<int NUM_COLS>
static __forceinline uint8_t compactCols(
    uint64_t* __restrict occ_word,
    uint64_t* __restrict color_words,   // boards_[0..kNumColors-1].lo or .hi
    int color_stride                    // byte stride between color words (= sizeof BitBoard = 16)
) {
    uint8_t fallen_mask = 0;
    for (int local = 0; local < NUM_COLS; ++local) {
        const int shift = local * config::Board::kBitsPerCol;

        const uint32_t occ_lane = static_cast<uint32_t>(*occ_word >> shift) & kColLaneMask;
        if (occ_lane == 0) continue;

        const int cnt = _mm_popcnt_u32(occ_lane);
        const uint32_t filled   = static_cast<uint32_t>(std::min(cnt, static_cast<int>(config::Board::kHeight)));
        const uint32_t new_occ  = (1u << filled) - 1u;

        if (occ_lane == new_occ) continue; // already compact

        const uint64_t clear = ~(static_cast<uint64_t>(kColLaneMask) << shift);

        for (int i = 0; i < config::Board::kNumColors; ++i) {
            uint64_t& cw = *reinterpret_cast<uint64_t*>(
                reinterpret_cast<char*>(color_words) + static_cast<size_t>(i) * static_cast<size_t>(color_stride));
            const uint32_t lane = static_cast<uint32_t>(cw >> shift) & kColLaneMask;
            if (lane == 0) continue;

            uint32_t compacted = _pext_u32(lane, occ_lane) & new_occ;
            if (compacted != lane) {
                fallen_mask |= (1 << i);
            }
            cw = (cw & clear) | (static_cast<uint64_t>(compacted) << shift);
        }

        *occ_word = (*occ_word & clear) | (static_cast<uint64_t>(new_occ) << shift);
    }
    return fallen_mask;
}

uint8_t Gravity::execute(Board& board) {
    uint8_t m1 = compactCols<config::Board::kColsInLo>(
        &board.occupancy_.lo,
        &board.boards_[0].lo,
        sizeof(BitBoard)
    );

    uint8_t m2 = compactCols<config::Board::kColsInHi>(
        &board.occupancy_.hi,
        &board.boards_[0].hi,
        sizeof(BitBoard)
    );

    return m1 | m2;
}

bool Gravity::canFall(const Board& board) {
    // A board can fall if there is any '1' bit (occupied) directly above a '0' bit (empty).
    // We can check all columns in each 64-bit word simultaneously using bitwise shifts.
    // Mask out bits that would shift across column boundaries (top row 15 of each 16-bit lane).
    static constexpr uint64_t kBoundaryMask = 0x8000800080008000ULL;

    const uint64_t lo = board.getOccupied().lo;
    // (word >> 1) moves bits from row y+1 to row y within the same column.
    // (shifted & ~word) has a bit set if row y+1 was 1 and row y was 0.
    if (((lo >> 1) & ~kBoundaryMask & ~lo) != 0) return true;

    const uint64_t hi = board.getOccupied().hi;
    if (((hi >> 1) & ~kBoundaryMask & ~hi) != 0) return true;

    return false;
}

} // namespace puyotan
