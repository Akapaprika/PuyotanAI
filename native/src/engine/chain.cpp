#include "engine/chain.hpp"
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <iostream>

#include "config/engine_config.hpp"
#include "engine/gravity.hpp"

namespace puyotan {

BitBoard Chain::findGroups(const BitBoard& color_board, int min_size) {
    BitBoard erased_total;
    BitBoard remaining = color_board;

    while (!remaining.empty()) {
        // Pick one bit to start a flood fill
        uint64_t start_lo = 0;
        uint64_t start_hi = 0;
        
        if (remaining.lo != 0) {
            start_lo = remaining.lo & -(int64_t)remaining.lo;
        } else {
            start_hi = remaining.hi & -(int64_t)remaining.hi;
        }

        BitBoard current_group(start_lo, start_hi);
        BitBoard last_group;

        // Flood fill: expand current_group until no new neighbors are found
        while (current_group != last_group) {
            last_group = current_group;
            // Expand in 4 directions
            BitBoard expanded = current_group;
            expanded |= current_group.shiftUp();
            expanded |= current_group.shiftDown();
            expanded |= current_group.shiftLeft();
            expanded |= current_group.shiftRight();
            
            // Limit expansion to the same color
            current_group = expanded & color_board;
        }

        // Count bits in the group
        int count = 0;
        // Optimization: Use __builtin_popcount or equivalent if available, 
        // but for now simple bit counting (popcount) is used.
        #ifdef _MSC_VER
            count = (int)(__popcnt64(current_group.lo) + __popcnt64(current_group.hi));
        #else
            count = __builtin_popcountll(current_group.lo) + __builtin_popcountll(current_group.hi);
        #endif

        if (count >= min_size) {
            erased_total |= current_group;
        }

        // Remove the processed group from remaining
        remaining &= ~current_group;
    }

    return erased_total;
}

bool Chain::execute(Board& board) {
    BitBoard total_erased_color;

    // 1. Find all color groups to erase
    // Rule::kColors is 4 (Red, Green, Blue, Yellow)
    for (int i = 0; i < config::Rule::kColors; ++i) {
        Cell c = static_cast<Cell>(i);
        BitBoard color_bb = board.getBitboard(c);
        if (color_bb.empty()) {
            continue;
        }

        BitBoard erased = findGroups(color_bb, config::Rule::kConnectCount);
        
        if (!erased.empty()) {
            total_erased_color |= erased;
            board.setBitboard(c, color_bb & ~erased);
        }
    }

    if (total_erased_color.empty()) {
        return false;
    }

    // 2. Erase adjacent Ojama puyos
    BitBoard ojama_bb = board.getBitboard(Cell::Ojama);
    if (!ojama_bb.empty()) {
        // Expand erased area by 1 to catch adjacent Ojamas
        BitBoard adjacent_mask;
        adjacent_mask |= total_erased_color.shiftUp();
        adjacent_mask |= total_erased_color.shiftDown();
        adjacent_mask |= total_erased_color.shiftLeft();
        adjacent_mask |= total_erased_color.shiftRight();

        BitBoard erased_ojama = ojama_bb & adjacent_mask;
        if (!erased_ojama.empty()) {
            board.setBitboard(Cell::Ojama, ojama_bb & ~erased_ojama);
        }
    }

    return true;
}

int Chain::executeChain(Board& board) {
    int chain_count = 0;
    
    // Initial gravity to settle anything floating
    Gravity::execute(board);

    while (execute(board)) {
        chain_count++;
        // Settle pieces after erasure
        Gravity::execute(board);
    }

    return chain_count;
}

} // namespace puyotan
