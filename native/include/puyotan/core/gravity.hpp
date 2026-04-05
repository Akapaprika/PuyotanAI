#pragma once

#include <puyotan/core/board.hpp>

namespace puyotan {
/**
 * @class Gravity
 * @brief Utility class for processing world-class bitwise gravity.
 *
 * Provides static methods to simulate vertical puyo falling.
 */
class Gravity {
  public:
    /**
     * @brief Drops all puyos until they hit the bottom or another puyo.
     * @param board The board to process.
     * @return A bitmask of all colors that actually moved during this step.
     * @note Performance: O(Columns) using SIMD/SWAR-PEXT compaction.
     */
    static uint32_t execute(Board& board) noexcept;

    /**
     * @brief Fast-check if any puyos are currently in a floating state.
     * @param board The board to check.
     * @return True if at least one puyo can fall.
     * @note Performance: O(1) using SIMD bit-shifting.
     */
    static bool canFall(const Board& board) noexcept;
};
} // namespace puyotan
