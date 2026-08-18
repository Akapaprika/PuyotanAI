#include <algorithm>
#include <cassert>
#include <immintrin.h>
#include <puyotan/core/board.hpp>

namespace puyotan {

Cell Board::get(int x, int y) const noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    // 16bitレーン直接アクセスにより、シフト量は単純な y のみで完結
    const int b1  = (boards_[1].cols[x] >> y) & 1;
    const int b2  = (boards_[2].cols[x] >> y) & 1;
    const int b3  = (boards_[3].cols[x] >> y) & 1;
    const int b4  = (boards_[4].cols[x] >> y) & 1;
    const int occ = (occupancy_.cols[x] >> y) & 1;

    // 分岐もループも使わない完全等価な状態方程式
    const int color = b1 + (b2 * 2) + (b3 * 3) + (b4 * 4) + (5 * (1 - occ));
    return static_cast<Cell>(color);
}

void Board::set(int x, int y, Cell color) noexcept {
    assert(color != Cell::Empty);
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    const uint16_t bit = static_cast<uint16_t>(1U << y);
    const uint16_t clear_mask = static_cast<uint16_t>(~bit);

    // 16bit単位の高速なクリア処理
    for (auto& bb : boards_) {
        bb.cols[x] &= clear_mask;
    }

    boards_[toIndex(color)].cols[x] |= bit;
    occupancy_.cols[x] |= bit;
}

void Board::clear(int x, int y) noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    // マスクを1回だけ生成して各色プレーンをクリア
    const uint16_t clear_mask = static_cast<uint16_t>(~(1U << y));
    for (auto& bb : boards_) {
        bb.cols[x] &= clear_mask;
    }
    occupancy_.cols[x] &= clear_mask;
}

void Board::placePiece(int col, Cell color) noexcept {
    assert(col >= 0 && col < config::Board::kWidth);
    set(col, config::Board::kSpawnRow, color);
}

void Board::setBitboard(Cell color, const BitBoard& bb) noexcept {
    boards_[toIndex(color)] = bb;
}

void Board::updateOccupancyFromBoards() noexcept {
    // 木構造リダクション (Tree-reduction) による並列OR結合
    // (0 | 1) と (2 | 3) を独立したSIMD実行ポートで並列実行し、レイテンシを短縮
    const __m128i or01   = _mm_or_si128(boards_[0].m128, boards_[1].m128);
    const __m128i or23   = _mm_or_si128(boards_[2].m128, boards_[3].m128);
    const __m128i or0123 = _mm_or_si128(or01, or23);
    occupancy_.m128      = _mm_or_si128(or0123, boards_[4].m128);
}

} // namespace puyotan