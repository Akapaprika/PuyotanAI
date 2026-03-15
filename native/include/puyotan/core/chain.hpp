#pragma once

#include <vector>
#include <puyotan/core/board.hpp>

namespace puyotan {

/**
 * ErasureData
 *   Pure data representing what was erased in a single chain step.
 */
struct ErasureData {
    bool erased = false;
    int num_erased = 0;
    int num_colors = 0;
    std::vector<int> group_sizes;
};

/**
 * Chain
 *   Handles piece erasure and chain detection logic.
 *   Purely mathematical/logical; scoring is handled by Scorer.
 */
class Chain {
public:
    /**
     * Finds and erases groups of connected puyos.
     * @param board The board to process.
     * @param color_mask Bitmask of colors to check.
     * @return ErasureData detailing the erasures in this step.
     */
    static ErasureData execute(Board& board, uint8_t color_mask = 0x0F);

    /**
     * Finds connected bits in a single color's BitBoard.
     */
    static BitBoard findGroups(const BitBoard& color_board, int min_size);
};

} // namespace puyotan
