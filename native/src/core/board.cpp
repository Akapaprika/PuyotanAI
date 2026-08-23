#include <bit>
#include <cassert>
#include <puyotan/core/board.hpp>

namespace puyotan {

// ★【最適化】実際のぷよ数（popcount）ピッタリで vector を一発確保
std::vector<Board::ActivePuyo> Board::getActivePuyos() const noexcept {
    std::vector<ActivePuyo> result;
    result.reserve(occupancy_.popcount());

    for (int ci = 0; ci < config::Board::kNumColors; ++ci) {
        const Cell color = static_cast<Cell>(ci);
        const BitBoard& bb = boards_[ci];
        for (int x = 0; x < config::Board::kWidth; ++x) {
            uint32_t col_bits = bb.cols[x] & config::Board::kVisibleColMask;
            while (col_bits) {
                const int y = std::countr_zero(col_bits);
                result.push_back({x, y, color});
                col_bits &= (col_bits - 1);
            }
        }
    }
    return result;
}

// ★【最適化】occupancy_ が 0 なら即 Cell::Empty をリターン（5色走査を完全スキップ）
Cell Board::get(int x, int y) const noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    const uint16_t mask = static_cast<uint16_t>(1U << y);
    if ((occupancy_.cols[x] & mask) == 0) {
        return Cell::Empty;
    }

    for (int ci = 0; ci < config::Board::kNumColors; ++ci) {
        if (boards_[ci].cols[x] & mask) {
            return static_cast<Cell>(ci);
        }
    }
    return Cell::Empty;
}

// ★【最適化】既に何か置かれていた場合のみ他色をクリア
void Board::set(int x, int y, Cell color) noexcept {
    assert(color != Cell::Empty);
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    const uint16_t bit = static_cast<uint16_t>(1U << y);
    const uint16_t clear_mask = static_cast<uint16_t>(~bit);

    if ((occupancy_.cols[x] & bit) != 0) {
        for (auto& bb : boards_) {
            bb.cols[x] &= clear_mask;
        }
    }

    boards_[toIndex(color)].cols[x] |= bit;
    occupancy_.cols[x] |= bit;
}

// ★【最適化】最初から空なら何もしない（無駄な 5色クリアを完全スキップ）
void Board::clear(int x, int y) noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);

    const uint16_t bit = static_cast<uint16_t>(1U << y);
    if ((occupancy_.cols[x] & bit) == 0) {
        return;
    }

    const uint16_t clear_mask = static_cast<uint16_t>(~bit);
    for (auto& bb : boards_) {
        bb.cols[x] &= clear_mask;
    }
    occupancy_.cols[x] &= clear_mask;
}

} // namespace puyotan