#pragma once

#include <array>
#include <cstdint>
#include <puyotan/core/board.hpp>

namespace puyotan {

/**
 * ErasureData
 *   Pure data representing what was erased in a single chain step.
 *   Uses a fixed-size stack array for group_sizes to avoid heap allocation.
 */
struct ErasureData {
    // erased == (num_erased > 0)
    int num_erased = 0;
    int num_colors = 0;
    uint8_t num_groups = 0;
    // Max possible groups in a 6x13 field with min-4 connect = floor(78/4) = 19.
    // We use 24 for alignment / safety headroom.
    std::array<uint8_t, 24> group_sizes{};
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
     * Checks if any groups can be erased.
     * @param board The board to check.
     * @return True if at least one group can be fired.
     */
    static bool canFire(const Board& board);
};

} // namespace puyotan
