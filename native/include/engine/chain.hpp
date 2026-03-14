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
    // @param color_mask Bitmask of colors to check (0x0F = all 4 colors).
    // @return true if any puyo (color or Ojama) was erased.
    static bool execute(Board& board, uint8_t color_mask = 0x0F);

    // Executes the full chain sequence:
    // Repeatedly applies Gravity -> Erase -> Gravity until the board is stable.
    //
    // @param board The board to process.
    // @param first_color_mask Optimization: check only these colors in the first pass.
    // @return The total number of chain steps executed (number of times erasures occurred).
    static int executeChain(Board& board, uint8_t first_color_mask = 0x0F);

    // Finds all groups in a single color's BitBoard that have at least
    // min_size connected bits.
    //
    // @param color_board The BitBoard of a single color.
    // @param min_size Minimum number of connected bits to be included.
    // @return A BitBoard containing only the bits belonging to valid groups.
    static BitBoard findGroups(const BitBoard& color_board, int min_size);
};

} // namespace puyotan
