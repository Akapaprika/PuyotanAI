#pragma once

#include <array>
#include <cstdint>
#include <puyotan/core/board.hpp>

namespace puyotan {
/**
 * @struct ErasureData
 * @brief Container for results of a puyo erasure scan.
 */
struct ErasureData {
    /// Sizes of each erased group (up to kMaxErasureGroups).
    std::array<uint8_t, config::Rule::kMaxErasureGroups> group_sizes;

    /// Bitmask of erased puyos for each color plane.
    std::array<BitBoard, config::Rule::kColors> erased_per_color;

    /// Combined bitmask of all erased puyos (including Ojama).
    BitBoard total_erased;

    int num_erased = 0; ///< Total number of non-ojama puyos erased
    int num_colors = 0; ///< Number of distinct colors erased (for score bonus)
    int num_groups = 0; ///< Total number of groups found
};

/**
 * @class Chain
 * @brief Logic for detecting and processing puyo erasures (chains).
 */
class Chain {
  public:
    static constexpr uint32_t kAllColorsMask = (1u << config::Rule::kColors) - 1u;

    /**
     * @brief Detects and applies erasures to the board in a single step.
     * @param board The board to process and modify.
     * @param color_mask Bitmask of which puyo colors to check for connections.
     * @return Data describing the erasures performed.
     */
    static ErasureData execute(Board& board, uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * @brief Scans for erasable groups WITHOUT modifying the board state.
     * @param board The board to scan.
     * @param color_mask Bitmask of colors to check.
     * @return ErasureData containing found groups and bitmasks.
     */
    static ErasureData findGroups(const Board& board, uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * @brief Commits pre-calculated erasure data to the board.
     * @param board The board to update.
     * @param data The result of a previous findGroups() call.
     */
    static void applyErasure(Board& board, const ErasureData& data) noexcept;

    /**
     * @brief Quick check if any groups are currently erasable.
     * @param board The board to check.
     * @param color_mask colors to consider.
     * @return True if at least one group will fire.
     */
    static bool canFire(const Board& board, uint32_t color_mask = kAllColorsMask) noexcept;
};
} // namespace puyotan
