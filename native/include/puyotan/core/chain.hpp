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
    // Per-color and total BitBoards of puyos to be erased.
    // Populated by findGroups(); used by applyErasure() to skip re-scanning.
    std::array<BitBoard, config::Rule::kColors> erased_per_color;
    BitBoard total_erased;
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
     * Scans for erasable groups WITHOUT modifying the board.
     * Stores per-color BitBoard masks and group sizes in the returned ErasureData.
     * Call applyErasure() in the next turn to commit the result.
     * @param board The board to scan.
     * @param color_mask Bitmask of colors to check.
     * @return ErasureData with group info and BitBoard masks (board unchanged).
     */
    static ErasureData findGroups(const Board& board, uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * Applies pre-computed erasure data to the board (no re-scan).
     * Call only after findGroups() on the same board state.
     * @param board The board to modify.
     * @param data The ErasureData returned by findGroups().
     */
    static void applyErasure(Board& board, const ErasureData& data) noexcept;

    static bool canFire(const Board& board, uint32_t color_mask = kAllColorsMask) noexcept;
};

} // namespace puyotan
