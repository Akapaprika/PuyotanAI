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
    BitBoard occupied = board.getOccupied();

    for (int step = 0; step < config::Board::kHeight; ++step) {
        // Build 'can_fall' mask: 
        // A puyo can fall if there is an empty space directly below it.
        const BitBoard can_fall = occupied & (~occupied & kFullMask).shiftUp();

        if (can_fall.empty()) {
            break;
        }

        moved = true;

        // Update combined occupancy incrementally.
        // Falling bits drop one row; staying bits are unchanged.
        occupied = (occupied & ~can_fall) | (occupied & can_fall).shiftDown();

        // Update individual color planes.
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const Cell c = static_cast<Cell>(i);
            const BitBoard bb = board.getBitboard(c);
            board.setBitboard(c, (bb & ~can_fall) | (bb & can_fall).shiftDown());
        }
    }

    // Update the board's permanent occupancy mask after settlement.
    // (setBitboard already updates it internally, but the loop settles it step-by-step).
    // Actually, board.setBitboard re-calculates it, which is safe but slightly slow 
    // inside the loop. Let's optimize setBitboard if needed later, but for now 
    // we must also handle the spawn-row erasure.

    // Erase any piece left in the invisible spawn row.
    const BitBoard keep = ~kSpawnRowMask;
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        Cell c = static_cast<Cell>(i);
        board.setBitboard(c, board.getBitboard(c) & keep);
    }

    return moved;
}

} // namespace puyotan
