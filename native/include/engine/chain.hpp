#pragma once

#include "engine/board.hpp"

namespace puyotan {

// ============================================================
// Chain
//   Handles puyo erasure and chain detection logic.
//
//   Puyotan β rules:
//     - 4 or more connected puyos of the same color are erased.
//     - Erasing color puyos also erases adjacent Ojama puyos.
// ============================================================
class Chain {
public:
    // Scans the board for any groups of connected puyos that meet the
    // minimum connection requirement (config::Rule::kConnectCount).
    // Erases those groups and any adjacent Ojama puyos.
    //
    // @param board The board to process.
    // @return true if any puyo (color or Ojama) was erased.
    static bool execute(Board& board);

    // Executes the full chain sequence:
    // Repeatedly applies Gravity -> Erase -> Gravity until the board is stable.
    //
    // @param board The board to process.
    // @return The total number of chain steps executed (number of times erasures occurred).
    static int executeChain(Board& board);

    // Finds all groups in a single color's BitBoard that have at least
    // min_size connected bits.
    //
    // @param color_board The BitBoard of a single color.
    // @param min_size Minimum number of connected bits to be included.
    // @return A BitBoard containing only the bits belonging to valid groups.
    static BitBoard findGroups(const BitBoard& color_board, int min_size);
};

} // namespace puyotan
