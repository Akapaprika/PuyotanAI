#include <puyotan/core/gravity.hpp>

namespace puyotan {

// Mask covering bits 0 through kTotalRows-1 (one column lane: 15 bits = rows 0-14)
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

// ---------------------------------------------------------------------------
// Per-column PEXT compaction helper.
// Called twice: once for lo-word columns (0-3), once for hi-word columns (4-5).
// Template on the word pointer avoids the use_hi branch inside the hot loop.
// ---------------------------------------------------------------------------
template<int NUM_COLS>
static __forceinline void compactCols(
    uint64_t* __restrict occ_word,
    uint64_t* __restrict color_words,   // boards_[0..kNumColors-1].lo or .hi
    int color_stride                    // byte stride between color words (= sizeof BitBoard = 16)
) {
    for (int local = 0; local < NUM_COLS; ++local) {
        const int shift = local * config::Board::kBitsPerCol;

        const uint32_t occ_lane = static_cast<uint32_t>(*occ_word >> shift) & kColLaneMask;
        if (occ_lane == 0) continue;

        const int cnt = _mm_popcnt_u32(occ_lane);
        // Cap at kHeight (13) — use unsigned compare to avoid signed branch.
        const uint32_t filled   = (static_cast<uint32_t>(cnt) > static_cast<uint32_t>(config::Board::kHeight))
                                  ? static_cast<uint32_t>(config::Board::kHeight)
                                  : static_cast<uint32_t>(cnt);
        const uint32_t new_occ  = (1u << filled) - 1u;

        if (occ_lane == new_occ) continue; // already compact

        const uint64_t clear = ~(static_cast<uint64_t>(kColLaneMask) << shift);

        // Compact each color lane using PEXT
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            // Advance by stride in bytes — each BitBoard is 16 bytes, .lo is at offset 0
            uint64_t& cw = *reinterpret_cast<uint64_t*>(
                reinterpret_cast<char*>(color_words) + static_cast<size_t>(i) * static_cast<size_t>(color_stride));
            const uint32_t lane = static_cast<uint32_t>(cw >> shift) & kColLaneMask;
            if (lane == 0) continue;

            uint32_t compacted = _pext_u32(lane, occ_lane) & new_occ;
            cw = (cw & clear) | (static_cast<uint64_t>(compacted) << shift);
        }

        *occ_word = (*occ_word & clear) | (static_cast<uint64_t>(new_occ) << shift);
    }
}

int Gravity::execute(Board& board) {
    // -----------------------------------------------------------------------
    // O(1)-per-column gravity using BMI2 PEXT.
    //
    // Split into two explicit loops (lo-word and hi-word) to eliminate the
    // use_hi branch that was previously evaluated per-iteration. The compiler
    // now resolves both word pointers at compile time for each loop.
    //
    // Layout:
    //   lo: cols 0-3 at offsets 0, 16, 32, 48 bits
    //   hi: cols 4-5 at offsets 0, 16 bits
    //
    // sizeof(BitBoard) == 16 bytes, .lo at offset 0 → stride between
    // consecutive color boards' .lo/.hi is 16 bytes (sizeof BitBoard).
    // -----------------------------------------------------------------------

    // Lo columns (0-3): operate on .lo of each board
    compactCols<config::Board::kColsInLo>(
        &board.occupancy_.lo,
        &board.boards_[0].lo,
        sizeof(BitBoard)
    );

    // Hi columns (4-5): operate on .hi of each board
    compactCols<config::Board::kColsInHi>(
        &board.occupancy_.hi,
        &board.boards_[0].hi,
        sizeof(BitBoard)
    );

    return 0; // PEXT gravity is non-iterative; step count not applicable.
}

} // namespace puyotan
