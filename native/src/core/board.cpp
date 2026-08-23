#include <bit>
#include <cassert>
#include <puyotan/core/board.hpp>

namespace puyotan {

std::vector<Board::ActivePuyo> Board::getActivePuyos() const noexcept {
    std::vector<ActivePuyo> result;
    result.reserve(config::Board::kWidth * config::Board::kHeight);

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

Cell Board::get(int x, int y) const noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);
    const int b1  = (boards_[1].cols[x] >> y) & 1;
    const int b2  = (boards_[2].cols[x] >> y) & 1;
    const int b3  = (boards_[3].cols[x] >> y) & 1;
    const int b4  = (boards_[4].cols[x] >> y) & 1;
    const int occ = (occupancy_.cols[x] >> y) & 1;
    return static_cast<Cell>(b1 + (b2 * 2) + (b3 * 3) + (b4 * 4) + (5 * (1 - occ)));
}

void Board::set(int x, int y, Cell color) noexcept {
    assert(color != Cell::Empty);
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);
    const uint16_t bit = static_cast<uint16_t>(1U << y);
    const uint16_t clear_mask = static_cast<uint16_t>(~bit);
    for (auto& bb : boards_) {
        bb.cols[x] &= clear_mask;
    }
    boards_[toIndex(color)].cols[x] |= bit;
    occupancy_.cols[x] |= bit;
}

void Board::clear(int x, int y) noexcept {
    assert(x >= 0 && x < config::Board::kWidth);
    assert(y >= 0 && y < config::Board::kHeight + 1);
    const uint16_t clear_mask = static_cast<uint16_t>(~(1U << y));
    for (auto& bb : boards_) {
        bb.cols[x] &= clear_mask;
    }
    occupancy_.cols[x] &= clear_mask;
}

} // namespace puyotan