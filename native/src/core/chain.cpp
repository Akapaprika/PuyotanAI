#include <puyotan/core/chain.hpp>

namespace puyotan {

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