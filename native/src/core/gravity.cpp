#include <puyotan/core/gravity.hpp>
#include <algorithm>

namespace puyotan {

// Mask covering bits 0 through kTotalRows-1 (one column lane: 15 bits = rows 0-14)
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

int Gravity::execute(Board& board) {
    // -----------------------------------------------------------------------
    // O(1)-per-column gravity using BMI2 PEXT.
    //
    // BitBoard layout: each column occupies kBitsPerCol=16 bits in a uint64_t.
    //   lo: cols 0-3,  col c at bit offset (c     * 16)
    //   hi: cols 4-5,  col c at bit offset ((c-4) * 16)
    //
    // Gravity semantics: puyos fall to row 0 (gravity = compact set bits to LSB),
    //   preserving relative vertical order within each column.
    //
    // PEXT(src, mask) extracts bits from src at positions where mask has 1s,
    // contiguously packed to the low end of the result.
    // Since 'mask' is the occupancy lane (all occupied rows), PEXT on each
    // color lane maps color-bits to their compacted output rows automatically.
    // -----------------------------------------------------------------------

    for (int col = 0; col < config::Board::kWidth; ++col) {
        const int local   = col & 3;                               // position in its 64-bit word
        const int shift   = local * config::Board::kBitsPerCol;    // bit offset within the word
        const bool use_hi = (col >= config::Board::kColsInLo);

        uint64_t& occ_word = use_hi ? board.occupancy_.hi
                                    : board.occupancy_.lo;
        const uint32_t occ_lane = static_cast<uint32_t>(occ_word >> shift) & kColLaneMask;

        if (occ_lane == 0) continue; // empty column — skip

        // Number of puyos that fit in the visible zone (cap at kHeight = 13)
        const int filled = std::min(
            static_cast<int>(_mm_popcnt_u32(occ_lane)),
            config::Board::kHeight
        );
        const uint32_t new_occ_lane = (1u << filled) - 1u; // compact bits 0..filled-1

        if (occ_lane == new_occ_lane) continue; // already settled — skip

        const uint64_t clear_mask = ~(static_cast<uint64_t>(kColLaneMask) << shift);

        // Compact each color's lane via PEXT, then clamp to visible rows
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            uint64_t& cw = use_hi ? board.boards_[i].hi
                                  : board.boards_[i].lo;
            const uint32_t color_lane = static_cast<uint32_t>(cw >> shift) & kColLaneMask;

            if (color_lane == 0) continue;

            // PEXT: for each set bit in occ_lane (top-to-bottom = MSB-to-LSB order
            // for gravity), extract the corresponding color bit and pack downward.
            uint32_t compacted = _pext_u32(color_lane, occ_lane);
            compacted &= new_occ_lane; // clamp: discard anything above visible rows

            cw = (cw & clear_mask) | (static_cast<uint64_t>(compacted) << shift);
        }

        // Update the occupancy lane for this column
        occ_word = (occ_word & clear_mask) | (static_cast<uint64_t>(new_occ_lane) << shift);
    }

    return 0; // PEXT gravity is non-iterative; step count not applicable.
}

} // namespace puyotan
