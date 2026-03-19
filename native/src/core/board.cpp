#include <puyotan/core/board.hpp>
#include <cassert>
#include <algorithm>

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
    assert(col >= 0 && col < config::Board::kWidth);
    set(col, config::Board::kSpawnRow, color);
}

int Board::getDropDistance(int x, int y) const {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y > 0 && y <= static_cast<int>(config::Board::kHeight));
    return y - getColumnHeight(x);
}

const BitBoard& Board::getBitboard(Cell color) const {
    return boards_[toIndex(color)];
}

void Board::setBitboard(Cell color, const BitBoard& bb, bool update_occupancy) {
    boards_[toIndex(color)] = bb;
    if (update_occupancy) {
        occupancy_ = boards_[0];
        for (int i = 1; i < config::Board::kNumColors; ++i) {
            occupancy_ |= boards_[i];
        }
    }
}

} // namespace puyotan
