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
    BitBoard total_erased;

    std::array<uint8_t, config::Rule::kMaxErasureGroups>
        group_sizes;    // 24バイト (オフセット 16〜39)
    int num_erased = 0; // 4バイト  (オフセット 40〜43)
    int num_colors = 0; // 4バイト  (オフセット 44〜47)
    int num_groups = 0; // 4バイト  (オフセット 48〜51)

    void clear() noexcept {
        total_erased = BitBoard();
        num_erased = 0;
        num_colors = 0;
        num_groups = 0;
    }
};

/**
 * @class Chain
 * @brief Logic for detecting and processing puyo erasures (chains).
 */
class Chain {
  public:
    static constexpr uint32_t kAllColorsMask =
        (1u << config::Rule::kColors) - 1u;

    /**
     * @brief Detects and applies erasures to the board in a single step.
     * @param board The board to process and modify.
     * @param color_mask Bitmask of which puyo colors to check for connections.
     * @return Data describing the erasures performed.
     */
    static ErasureData execute(Board& board,
                               uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * @brief Scans for erasable groups WITHOUT modifying the board state.
     * @param board The board to scan.
     * @param color_mask Bitmask of colors to check.
     * @return ErasureData containing found groups and bitmasks.
     */
    static void scanGroups(const Board& board, ErasureData& erasure_data,
                           uint32_t color_mask = kAllColorsMask) noexcept;

    /**
     * @brief Commits pre-calculated erasure data to the board.
     * @param board The board to update.
     * @param data The result of a previous scanGroups() call.
     */
    static void applyErasure(Board& board, const ErasureData& data) noexcept;

    /**
     * @brief Quick check if any groups are currently erasable.
     * @param board The board to check.
     * @param color_mask colors to consider.
     * @return True if at least one group will fire.
     */
    static bool canFire(const Board& board,
                        uint32_t color_mask = kAllColorsMask) noexcept;
};
} // namespace puyotan
