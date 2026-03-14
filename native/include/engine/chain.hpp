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
    struct StepResult {
        bool erased = false;
        int num_erased = 0;
        int num_colors = 0;
        int group_bonus = 0;
        int score = 0;
    };

    struct ChainResult {
        int total_score = 0;
        int max_chain = 0;
        int total_erased = 0;
    };

    // Scans the board for any groups of connected puyos that meet the
    // minimum connection requirement (config::Rule::kConnectCount).
    // Erases those groups and any adjacent Ojama puyos.
    //
    // @param board The board to process.
    // @param chain_number Current chain step (1-indexed) for bonus calculation.
    // @param color_mask Bitmask of colors to check (0x0F = all 4 colors).
    // @return StepResult containing score and erasure info.
    static StepResult execute(Board& board, int chain_number, uint8_t color_mask = 0x0F);

    // Executes the full chain sequence:
    // Repeatedly applies Gravity -> Erase -> Gravity until the board is stable.
    //
    // @param board The board to process.
    // @param first_color_mask Optimization: check only these colors in the first pass.
    // @return ChainResult containing total score and chain count.
    static ChainResult executeChain(Board& board, uint8_t first_color_mask = 0x0F);

    // Finds all groups in a single color's BitBoard that have at least
    // min_size connected bits.
    //
    // @param color_board The BitBoard of a single color.
    // @param min_size Minimum number of connected bits to be included.
    // @return A BitBoard containing only the bits belonging to valid groups.
    static BitBoard findGroups(const BitBoard& color_board, int min_size);
};

} // namespace puyotan
