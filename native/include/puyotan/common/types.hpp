#pragma once

#include <cstdint>

namespace puyotan {

/**
 * Cell types in the Puyo Puyo board.
 */
enum class Cell : int {
    Red = 0,
    Green = 1,
    Blue = 2,
    Yellow = 3,
    Ojama = 4,
    Empty = 5
};

/**
 * Rotation of the sub puyo relative to the axis puyo.
 */
enum class Rotation : int {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3
};

/**
 * A pair of puyos (tsumo piece).
 */
struct PuyoPiece {
    Cell axis;
    Cell sub;
};

} // namespace puyotan
