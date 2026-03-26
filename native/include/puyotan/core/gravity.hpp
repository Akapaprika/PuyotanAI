#pragma once

#include <puyotan/core/board.hpp>

namespace puyotan {

/**
 * Gravity
 *   Utility for processing world-class bitwise gravity.
 */
class Gravity {
public:
    /**
     * Drops all puyos until they hit the bottom or another puyo.
     * @return A bitmask of all colors that actually fell.
     */
    static uint32_t execute(Board& board) noexcept;

    /**
     * Checks if any puyos can fall.
     * @param board The board to check.
     * @return True if at least one puyo can fall.
     */
    static bool canFall(const Board& board) noexcept;
};

} // namespace puyotan
