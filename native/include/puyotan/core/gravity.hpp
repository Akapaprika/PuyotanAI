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
     * @return The number of steps (rows) shifted.
     */
    static int execute(Board& board);
};

} // namespace puyotan
