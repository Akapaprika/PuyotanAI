#include "engine/gravity.hpp"
#include "config/engine_config.hpp"

namespace puyotan {

// Mask covering the invisible spawn row (row 13) in all columns.
// Any puyo remaining here after gravity must be discarded (Puyotan β rule).
static constexpr BitBoard kSpawnRowMask {
    config::Board::kLoSpawnMask,
    config::Board::kHiSpawnMask
};

// Full valid-cell mask (rows 0-13, all columns).
static constexpr BitBoard kFullMask {
    config::Board::kLoMask,
    config::Board::kHiMask
};

// ============================================================
// Gravity::execute
//   Applies gravity to every color plane simultaneously using
//   bit-shift operations, then erases any piece left in the
//   invisible spawn row (Puyotan β rule).
//
//   Algorithm (one step):
//     1. Build "occupied" = union of all color planes.
//     2. "empty" = complement of occupied, masked to valid cells.
//     3. "can_fall" = pieces that have an empty cell directly below,
//               computed as: occupied & empty.shift_up()
//                            (shift empty UP = expose the row above each gap)
//     4. For each color: falling bits shift_down(); staying bits are kept.
//
//   We repeat up to kHeight times (worst case: one puyo drops 13 rows).
//   In practice the loop exits early via the empty() check.
// ============================================================
bool Gravity::execute(Board& board) {
    bool moved = false;

    for (int step = 0; step < config::Board::kHeight; ++step) {
        // ---- Build combined occupancy (no switch: loop over all planes) ----
        BitBoard occupied;
        for (int i = 0; i < kNumCellColors; ++i)
            occupied |= board.get_bitboard(static_cast<Cell>(i + 1));

        // ---- Identify pieces that can fall one row ----
        // empty & kFullMask because ~occupied has garbage in unused bits
        BitBoard empty = (~occupied) & kFullMask;
        BitBoard can_fall = occupied & empty.shift_up();
        // (shift_up moves the empty-space bits toward higher rows,
        //  so the intersection finds pieces sitting above a gap)

        if (can_fall.empty()) break; // nothing moves → done

        moved = true;

        // ---- Apply fall: per color, branchless on the BitBoard level ----
        for (int i = 0; i < kNumCellColors; ++i) {
            Cell c = static_cast<Cell>(i + 1);
            BitBoard bb = board.get_bitboard(c);
            // Bits in can_fall drop one row; the rest stay.
            board.set_bitboard(c, (bb & ~can_fall) | (bb & can_fall).shift_down());
        }
    }

    // ---- Puyotan β: erase spawn row after every gravity pass ----
    // Pieces that couldn't fall (column full) would sit here permanently
    // without this clear – which matches the JS reference behaviour.
    BitBoard keep = ~kSpawnRowMask; // complement = everything except spawn row
    for (int i = 0; i < kNumCellColors; ++i) {
        Cell c = static_cast<Cell>(i + 1);
        board.set_bitboard(c, board.get_bitboard(c) & keep);
    }

    return moved;
}

} // namespace puyotan
