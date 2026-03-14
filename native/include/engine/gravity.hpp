#pragma once

#include "engine/board.hpp"

namespace puyotan {

class Gravity {
public:
    // Apply gravity to the entire board, making floating puyos fall.
    // Returns true if any puyo moved, false if the board state was unchanged.
    static bool execute(Board& board);
};

} // namespace puyotan
