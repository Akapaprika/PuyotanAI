#pragma once

#include <array>
#include <cstdint>

namespace puyotan {
/**
 * @enum Cell
 * @brief Represents the type of a single grid cell on the Puyo Puyo board.
 */
enum class Cell : uint8_t {
    Red = 0,    ///< Red Puyo
    Green = 1,  ///< Green Puyo
    Blue = 2,   ///< Blue Puyo
    Yellow = 3, ///< Yellow Puyo
    Ojama = 4,  ///< Nuisance (Ojama) Puyo
    Empty = 5   ///< Empty space
};

/**
 * @enum Rotation
 * @brief Specifies the orientation of the sub-puyo relative to the axis-puyo.
 */
enum class Rotation : uint8_t {
    Up = 0,    ///< Sub-puyo is above the axis
    Right = 1, ///< Sub-puyo is to the right of the axis
    Down = 2,  ///< Sub-puyo is below the axis
    Left = 3   ///< Sub-puyo is to the left of the axis
};

/**
 * @struct PuyoPiece
 * @brief Represents a pair of puyos (Axis and Sub) that fall as a single piece.
 */
struct PuyoPiece {
    Cell axis;          ///< The primary puyo used for positioning reference
    Cell sub;           ///< The secondary puyo that rotates around the axis
    uint8_t dirty_flag; ///< Precomputed bitmask: (1 << axis) | (1 << sub)
    uint8_t pad;        ///< 4-byte padding alignment for structure stability
};

/**
 * @enum ActionType
 * @brief Defines the different states or commands handled by the frame-based engine.
 */
enum class ActionType : uint8_t {
    None = 0,      ///< Decision point: Player must provide an instruction
    Pass = 1,      ///< Player skips the turn (typically end-of-game)
    Put = 2,       ///< Piece placement action
    Chain = 3,     ///< Internal: Puyo connection/erasure processing
    ChainFall = 4, ///< Internal: Puyo gravity fall after erasure
    Ojama = 5      ///< Internal: Nuisance puyo falling process
};

/**
 * @struct Action
 * @brief A high-level command from a player (typically for 'Put' actions).
 */
struct Action {
    ActionType type = ActionType::None; ///< The type of command
    int8_t x = 0;                       ///< target column index (0-5)
    Rotation rotation = Rotation::Up;   ///< target rotation of the piece
};

/**
 * @struct ActionState
 * @brief Tracks an action's progress across multiple frames.
 */
struct ActionState {
    Action action;               ///< The action being executed
    uint8_t remaining_frame = 0; ///< Number of frames until action completion (0 or 1)
};

/**
 * @enum MatchStatus
 * @brief Final game results or current engine lifecycle state.
 */
enum class MatchStatus : uint8_t {
    Ready = 0,   ///< Initial state before the first frame starts
    Playing = 1, ///< Active gameplay or simulation
    WinP1 = 2,   ///< Player 1 has won the match
    WinP2 = 3,   ///< Player 2 has won the match
    Draw = 4     ///< Both players died simultaneously
};

/**
 * @brief Y-axis offset for the Axis puyo placement based on rotation.
 */
inline constexpr std::array<int8_t, 4> kAxisDy = {0, 0, 1, 0};

/**
 * @brief X-axis offset for the Sub puyo placement relative to the Axis.
 */
inline constexpr std::array<int8_t, 4> kSubDx = {0, 1, 0, -1};

/**
 * @brief Y-axis offset for the Sub puyo placement relative to the Axis.
 */
inline constexpr std::array<int8_t, 4> kSubDy = {1, 0, -1, 0};

/**
 * @brief Simplified Y-axis offset for Sub puyo (legacy/simple placement rules).
 */
inline constexpr std::array<int8_t, 4> kSubDySimple = {1, 0, 0, 0};
} // namespace puyotan
