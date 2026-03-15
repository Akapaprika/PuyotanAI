#include <puyotan/core/board.hpp>

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
    if (col < 0 || col >= config::Board::kWidth) {
        return;
    }
    set(col, config::Board::kSpawnRow, color);
}

int Board::getDropDistance(int x, int y) const {
    if (x < 0 || x >= config::Board::kWidth || y <= 0) {
        return 0;
    }
    
    // Pieces are only considered to "fall" within the first 13 rows.
    // Row 14 (sub-puyo in Up rotation) is treated as falling from row 13.
    int start_y = (y > config::Board::kHeight) ? config::Board::kHeight : y;
    
    // Check occupancy in column x, rows 0 to start_y-1
    uint64_t column_lane;
    if (x < config::Board::kColsInLo) {
        column_lane = (occupancy_.lo >> (x * config::Board::kBitsPerCol));
    } else {
        column_lane = (occupancy_.hi >> ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol));
    }
    
    // Mask bits below start_y: (1 << start_y) - 1
    uint64_t below_mask = (1ULL << start_y) - 1;
    uint64_t obstacles = column_lane & below_mask;
    
    if (obstacles == 0) {
        return start_y;
    }
    
    // Most significant bit of obstacles is the highest occupied row
    #ifdef _MSC_VER
        unsigned long highest_row;
        if (_BitScanReverse64(&highest_row, obstacles)) {
            return start_y - (int)highest_row - 1;
        }
        return start_y;
    #else
        int highest_row = 63 - __builtin_clzll(obstacles);
        return start_y - highest_row - 1;
    #endif
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
