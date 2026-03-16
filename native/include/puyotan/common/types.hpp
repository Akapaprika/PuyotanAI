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

/**
 * Action types for Puyotan frame-based management.
 */
enum class ActionType : int {
    PASS = 0,
    PUT = 1,
    CHAIN = 2,
    CHAIN_FALL = 3,
    OJAMA = 4
};

/**
 * Action command from player.
 */
struct Action {
    ActionType type = ActionType::PASS;
    int x = 0;
    Rotation rotation = Rotation::Up;
};

/**
 * Action state in a specific frame.
 */
struct ActionState {
    Action action;
    int remaining_frame = 0;
};

/**
 * Status of the match.
 */
enum class MatchStatus : int {
    READY = 0,
    PLAYING = 1,
    WIN_P1 = 2,
    WIN_P2 = 3,
    DRAW = 4
};

} // namespace puyotan
