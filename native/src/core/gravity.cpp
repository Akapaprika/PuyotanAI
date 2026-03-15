#include <puyotan/core/gravity.hpp>

namespace puyotan {

namespace {

// Mask covering only row 13 (spawn row)
BitBoard kSpawnRowMask() {
    return { config::Board::kLoSpawnMask, config::Board::kHiSpawnMask };
}

} // namespace

int Gravity::execute(Board& board) {
    int steps = 0;
    BitBoard occupied = board.getOccupied();

    for (int s = 0; s < config::Board::kTotalRows; ++s) {
        const BitBoard can_fall = occupied & (~occupied & BitBoard::kFullMask()).shiftUp();

        if (can_fall.empty()) {
            break;
        }

        ++steps;
        occupied = (occupied & ~can_fall) | (occupied & can_fall).shiftDown();

        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const Cell c = static_cast<Cell>(i);
            const BitBoard bb = board.getBitboard(c);
            board.setBitboard(c, (bb & ~can_fall) | (bb & can_fall).shiftDown(), false);
        }
        board.updateOccupancy(occupied);
    }

    // Clear rows 13 and above (spawn row and buffer).
    static const BitBoard kKeepMask{
        config::Board::kLoVisibleMask,
        config::Board::kHiVisibleMask
    };
    
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        const Cell c = static_cast<Cell>(i);
        board.setBitboard(c, board.getBitboard(c) & kKeepMask, false);
    }
    board.updateOccupancy(board.getOccupied() & kKeepMask);

    return steps;
}

} // namespace puyotan
