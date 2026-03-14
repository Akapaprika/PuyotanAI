#include "engine/board.hpp"

namespace puyotan {

Cell Board::get(int x, int y) const {
    if (!occupancy_.get(x, y)) {
        return Cell::Empty;
    }
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        if (boards_[i].get(x, y)) {
            return static_cast<Cell>(i);
        }
    }
    return Cell::Empty;
}

void Board::set(int x, int y, Cell color) {
    clear(x, y);
    if (color != Cell::Empty) {
        boards_[toIndex(color)].set(x, y);
        occupancy_.set(x, y);
    }
}

void Board::clear(int x, int y) {
    for (auto& bb : boards_) {
        bb.clear(x, y);
    }
    occupancy_.clear(x, y);
}

void Board::placePiece(int col, Cell color) {
    if (static_cast<unsigned>(col) < static_cast<unsigned>(config::Board::kWidth)) {
        set(col, config::Board::kSpawnRow, color);
    }
}

const BitBoard& Board::getBitboard(Cell color) const {
    return boards_[toIndex(color)];
}

void Board::setBitboard(Cell color, const BitBoard& bb) {
    boards_[toIndex(color)] = bb;
    
    // Re-calculate combined occupancy. 
    // This is occasionally called (e.g. at the end of setBitboard batch), 
    // so we re-OR everything.
    occupancy_ = BitBoard{};
    for (const auto& plane : boards_) {
        occupancy_ |= plane;
    }
}

} // namespace puyotan
