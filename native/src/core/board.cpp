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

    int start_y = std::min(y, (int)config::Board::kHeight);

    const uint64_t lanes[2] = { occupancy_.lo, occupancy_.hi };

    int idx = (x >= config::Board::kColsInLo);
    int shift = (x - idx * config::Board::kColsInLo) * config::Board::kBitsPerCol;

    uint64_t column_lane = lanes[idx] >> shift;

    uint64_t below_mask = (1ULL << start_y) - 1ULL;
    uint64_t obstacles = column_lane & below_mask;

    if (obstacles == 0) {
        return start_y;
    }

#ifdef _MSC_VER
    unsigned long highest_row;
    _BitScanReverse64(&highest_row, obstacles);
#else
    int highest_row = 63 - __builtin_clzll(obstacles);
#endif

    return start_y - (int)highest_row - 1;
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
