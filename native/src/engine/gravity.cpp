#include "engine/gravity.hpp"
#include "config/engine_config.hpp"

namespace puyotan {

// Spawn-row mask: bits for y=kSpawnRow in every column.
// After every gravity pass, anything left here is erased (Puyotan β rule).
static constexpr BitBoard kSpawnRowMask {
    config::Board::kLoSpawnMask,
    config::Board::kHiSpawnMask
};

// Full valid-cell mask (rows 0-13, all 6 columns).
static constexpr BitBoard kFullMask {
    config::Board::kLoMask,
    config::Board::kHiMask
};

bool Gravity::execute(Board& board) {
    bool moved = false;

    for (int step = 0; step < config::Board::kHeight; ++step) {
        // Build combined occupancy by OR-ing all color planes.
        BitBoard occupied;
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            occupied |= board.getBitboard(static_cast<Cell>(i));
        }

        // ~occupied has garbage in unused bits → mask to valid cells.
        BitBoard can_fall = occupied & (~occupied & kFullMask).shiftUp();

        if (can_fall.empty()) {
            break;
        }

        moved = true;

        // Falling bits drop one row; staying bits are unchanged.
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            Cell c = static_cast<Cell>(i);
            BitBoard bb = board.getBitboard(c);
            board.setBitboard(c, (bb & ~can_fall) | (bb & can_fall).shiftDown());
        }
    }

    // Erase any piece left in the invisible spawn row.
    const BitBoard keep = ~kSpawnRowMask;
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        Cell c = static_cast<Cell>(i);
        board.setBitboard(c, board.getBitboard(c) & keep);
    }

    return moved;
}

} // namespace puyotan
