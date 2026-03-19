#pragma once

#include <cstdint>
#include <array>

namespace puyotan {

/**
 * Cell types in the Puyo Puyo board.
 */
enum class Cell : uint8_t {
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
enum class Rotation : uint8_t {
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
enum class ActionType : uint8_t {
    NONE = 0,
    PASS = 1,
    PUT = 2,
    CHAIN = 3,
    CHAIN_FALL = 4,
    OJAMA = 5
};

/**
 * Action command from player.
 */
struct Action {
    ActionType type = ActionType::NONE;
    int8_t x = 0;           // 0..5 (board width)
    Rotation rotation = Rotation::Up;
};

/**
 * Action state in a specific frame.
 */
struct ActionState {
    Action action;
    uint8_t remaining_frame = 0; // always 0 or 1
};

/**
 * Status of the match.
 */
enum class MatchStatus : uint8_t {
    READY = 0,
    PLAYING = 1,
    WIN_P1 = 2,
    WIN_P2 = 3,
    DRAW = 4
};

// ---------------------------------------------------------------------------
// Shared piece-placement LUT — indexed by Rotation (Up=0 Right=1 Down=2 Left=3).
// Axis: placed at (x, col_height + kAxisDy[r]).
// Sub:  placed at (x + kSubDx[r], ...).
// ---------------------------------------------------------------------------
inline constexpr std::array<int8_t, 4> kAxisDy = { 0,  0,  1,  0 };
inline constexpr std::array<int8_t, 4> kSubDx  = { 0,  1,  0, -1 };
inline constexpr std::array<int8_t, 4> kSubDy  = { 1,  0, -1,  0 };
inline constexpr std::array<int8_t, 4> kSubDy_Simple = { 1,  0,  0,  0 };

} // namespace puyotan
