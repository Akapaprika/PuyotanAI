#pragma once

#include "engine/board.hpp"

namespace puyotan {

// ============================================================
// Gravity
//   Applies standard Puyotan β gravity to a Board.
//
//   Algorithm overview (execute):
//     Repeats up to kHeight times (worst-case a piece drops the
//     full board height):
//       1. Build "occupied" = OR of all 5 color BitBoard planes.
//       2. "empty" = ~occupied, masked to the 6×14 valid area.
//       3. "can_fall" = occupied & empty.shift_up()
//            (shift_up maps each empty cell to the cell above it;
//             AND with occupied finds pieces sitting above a gap.)
//       4. For each color plane: falling bits move down one row;
//          staying bits are unchanged.
//       5. Early-exit when can_fall is empty (nothing moves).
//
//     After the loop, any piece remaining in the invisible spawn
//     row (y = kSpawnRow) is erased — Puyotan β rule matching the
//     JS reference implementation (Puyo.js Field::fall()).
//
//   Returns:
//     true  — at least one piece moved (useful for chain detection).
//     false — the board was already stable.
//
//   Performance notes:
//     All operations are bitwise on two 64-bit words; there is no
//     per-cell branch inside the inner loop.  The outer loop runs
//     at most kHeight (= 13) iterations, typically far fewer.
// ============================================================
class Gravity {
public:
    /// Applies gravity and cleans up the spawn row.
    /// Returns the total distance fallen across all puyos in all columns.
    static int execute(Board& board);
};

} // namespace puyotan
