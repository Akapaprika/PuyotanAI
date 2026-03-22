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
    // Optimized: leave uninitialized as it is always accessed via num_groups.
    std::array<uint8_t, config::Rule::kMaxErasureGroups> group_sizes;
    int num_erased = 0;
    int num_colors = 0;
    int num_groups = 0;
};

/**
 * Chain
 *   Handles piece erasure and chain detection logic.
 *   Purely mathematical/logical; scoring is handled by Scorer.
 */
class Chain {
public:
    static constexpr uint32_t kAllColorsMask = (1u << config::Rule::kColors) - 1u;

    /**
     * Finds and erases groups of connected puyos.
     * @param board The board to process.
     * @param color_mask Bitmask of colors to check.
     * @return ErasureData detailing the erasures in this step.
     */
    static ErasureData execute(Board& board, uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * Checks if any groups can be erased.
     * @param board The board to check.
     * @return True if at least one group can be fired.
     */
    static bool canFire(const Board& board, uint32_t color_mask = kAllColorsMask) noexcept;
};

} // namespace puyotan
