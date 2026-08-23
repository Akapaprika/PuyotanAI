#include <immintrin.h>
#include <puyotan/core/chain.hpp>

namespace puyotan {

void Chain::applyErasure(Board& board, const ErasureData& data) noexcept {
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        board.boards_[i].andNot(data.total_erased);
    }
    board.occupancy_.andNot(data.total_erased);
}

ErasureData Chain::execute(Board& board, uint32_t color_mask) noexcept {
    ErasureData data;
    execute(board, data, color_mask);
    return data;
}

void Chain::execute(Board& board, ErasureData& data,
                    uint32_t color_mask) noexcept {
    scanGroups(board, data, color_mask);
    if (data.num_erased > 0) {
        applyErasure(board, data);
    }
}

} // namespace puyotan