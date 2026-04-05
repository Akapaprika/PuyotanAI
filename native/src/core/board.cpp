#include <algorithm>
#include <cassert>
#include <puyotan/core/board.hpp>

namespace puyotan {
Cell Board::get(int x, int y) const noexcept {
    if (!occupancy_.get(x, y)) {
        return Cell::Empty;
    }

    // SNEAKY OPTIMIZATION: This branchless implementation relies on the specific order
    // of Cell enum (0, 1, 2, 3, 4). Each color's bit contribution is multiplied by its index.
    // This avoids a 5-way conditional branch or map lookup.
    static_assert(static_cast<int>(Cell::Red) == 0);
    static_assert(static_cast<int>(Cell::Green) == 1);
    static_assert(static_cast<int>(Cell::Blue) == 2);
    static_assert(static_cast<int>(Cell::Yellow) == 3);
    static_assert(static_cast<int>(Cell::Ojama) == 4);
    static_assert(config::Board::kNumColors == 5);

    int found_index = 0;
    for (int i = 1; i < config::Board::kNumColors; ++i) {
        found_index += boards_[i].get(x, y) * i;
    }
    return static_cast<Cell>(found_index);
}

void Board::set(int x, int y, Cell color) noexcept {
    assert(color != Cell::Empty);
    clear(x, y);
    boards_[toIndex(color)].set(x, y);
    occupancy_.set(x, y);
}

void Board::clear(int x, int y) noexcept {
    for (auto& bb : boards_) {
        bb.clear(x, y);
    }
    occupancy_.clear(x, y);
}

void Board::placePiece(int col, Cell color) noexcept {
    assert(col >= 0 && col < config::Board::kWidth);
    set(col, config::Board::kSpawnRow, color);
}

int Board::getDropDistance(int x, int y) const noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y > 0 && y <= static_cast<int>(config::Board::kHeight));
    // Implementation note: This assumes 13th row (spawn) and visible field are contiguous.
    // Distance = current Y - top of existing stack.
    return y - getColumnHeight(x);
}

const BitBoard& Board::getBitboard(Cell color) const noexcept {
    return boards_[toIndex(color)];
}

void Board::setBitboard(Cell color, const BitBoard& bb) noexcept {
    boards_[toIndex(color)] = bb;
}

void Board::updateOccupancyFromBoards() noexcept {
    occupancy_ = boards_[0];
    for (int i = 1; i < config::Board::kNumColors; ++i) {
        occupancy_ |= boards_[i];
    }
}
} // namespace puyotan
