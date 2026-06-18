#include <algorithm>
#include <cassert>
#include <puyotan/core/board.hpp>

namespace puyotan {
Cell Board::get(int x, int y) const noexcept {
    const int idx = x >> 2;
    const int shift = ((x & 3) << 4) | y;

    // Red(0) のロードをスキップし、他の色と占有状況のみをロード
    const int b1 = static_cast<int>(((&boards_[1].lo)[idx] >> shift) & 1);
    const int b2 = static_cast<int>(((&boards_[2].lo)[idx] >> shift) & 1);
    const int b3 = static_cast<int>(((&boards_[3].lo)[idx] >> shift) & 1);
    const int b4 = static_cast<int>(((&boards_[4].lo)[idx] >> shift) & 1);
    const int occ = static_cast<int>(((&occupancy_.lo)[idx] >> shift) & 1);

    // 分岐もループも使わない完全等価な状態方程式
    // - 空 (occ=0) なら、式は 5 * (1 - 0) = 5 (Cell::Empty) となる
    // - 赤 (occ=1, 他が0) なら、式は 0 + 5 * 0 = 0 (Cell::Red) となる
    // - 他の色 (occ=1) なら、各ビットに対応するインデックス (1〜4) に収束する
    const int color = (b1 * 1) + (b2 * 2) + (b3 * 3) + (b4 * 4) + (5 * (1 - occ));
    return static_cast<Cell>(color);
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
