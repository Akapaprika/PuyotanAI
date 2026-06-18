#include <algorithm>
#include <cassert>
#include <puyotan/core/board.hpp>

namespace puyotan {
Cell Board::get(int x, int y) const noexcept {
    const int idx = x >> 2;
    const int shift = ((x & 3) << 4) | y;

    // 1. occupancy のチェック
    const uint64_t* occ_ptr = reinterpret_cast<const uint64_t*>(&occupancy_);
    if (!((occ_ptr[idx] >> shift) & 1)) {
        return Cell::Empty;
    }

    // 2.
    // 盤面が存在する場合、ループ内からは条件分岐（if）を完全に追放（ブランチレス化）します。
    int found_index = 0;
    for (int i = 1; i < config::Board::kNumColors; ++i) {
        const uint64_t* board_ptr =
            reinterpret_cast<const uint64_t*>(&boards_[i]);
        int bit = static_cast<int>((board_ptr[idx] >> shift) & 1);
        found_index |= (i & -bit);
    }

    return static_cast<Cell>(found_index);
}

void Board::set(int x, int y, Cell color) noexcept {
    assert(color != Cell::Empty);
    const int idx = x >> 2;
    const int shift = ((x & 3) << 4) | y;
    const uint64_t bit = 1ULL << shift;
    const uint64_t clear_mask = ~bit;

    // 共用体の構成要素である lo のポインタを経由させることで、
    // キャストを使わず安全かつ高速に連続メモリアクセスを行います。
    for (auto& bb : boards_) {
        (&bb.lo)[idx] &= clear_mask;
    }

    (&boards_[toIndex(color)].lo)[idx] |= bit;
    (&occupancy_.lo)[idx] |= bit;
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
    // Implementation note: This assumes 13th row (spawn) and visible field are
    // contiguous. Distance = current Y - top of existing stack.
    return y - getColumnHeight(x);
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
