#include "engine/chain.hpp"
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "config/engine_config.hpp"
#include "engine/gravity.hpp"
#include <algorithm>

namespace puyotan {

BitBoard Chain::findGroups(const BitBoard& color_board, int min_size) {
    BitBoard erased_total;
    BitBoard remaining = color_board;

    while (!remaining.empty()) {
        uint64_t start_lo = 0;
        uint64_t start_hi = 0;
        
        if (remaining.lo != 0) {
            start_lo = remaining.lo & -(int64_t)remaining.lo;
        } else {
            start_hi = remaining.hi & -(int64_t)remaining.hi;
        }

        BitBoard current_group(start_lo, start_hi);
        BitBoard last_group;

        while (current_group != last_group) {
            last_group = current_group;
            BitBoard expanded = current_group;
            expanded |= current_group.shiftUp();
            expanded |= current_group.shiftDown();
            expanded |= current_group.shiftLeft();
            expanded |= current_group.shiftRight();
            current_group = expanded & color_board;
        }

        int count = 0;
        #ifdef _MSC_VER
            count = (int)(__popcnt64(current_group.lo) + __popcnt64(current_group.hi));
        #else
            count = __builtin_popcountll(current_group.lo) + __builtin_popcountll(current_group.hi);
        #endif

        if (count >= min_size) {
            erased_total |= current_group;
        }

        remaining &= ~current_group;
    }

    return erased_total;
}

Chain::StepResult Chain::execute(Board& board, int chain_number, uint8_t color_mask) {
    StepResult result;
    BitBoard total_erased_color;

    // 1. Find all color groups to erase
    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!(color_mask & (1 << i))) {
            continue;
        }

        const Cell c = static_cast<Cell>(i);
        const BitBoard color_bb = board.getBitboard(c);
        if (color_bb.empty()) {
            continue;
        }

        // We need to call findGroups repeatedly or get counts per group for bonuses
        // For simplicity, we'll re-implement a bit of findGroups logic here to get per-group bonuses
        BitBoard remaining = color_bb;
        bool color_erased = false;

        while (!remaining.empty()) {
            uint64_t start_lo = 0;
            uint64_t start_hi = 0;
            if (remaining.lo != 0) start_lo = remaining.lo & -(int64_t)remaining.lo;
            else start_hi = remaining.hi & -(int64_t)remaining.hi;

            BitBoard current_group(start_lo, start_hi);
            BitBoard last_group;
            while (current_group != last_group) {
                last_group = current_group;
                BitBoard expanded = current_group;
                expanded |= current_group.shiftUp();
                expanded |= current_group.shiftDown();
                expanded |= current_group.shiftLeft();
                expanded |= current_group.shiftRight();
                current_group = expanded & color_bb;
            }

            int count = 0;
            #ifdef _MSC_VER
                count = (int)(__popcnt64(current_group.lo) + __popcnt64(current_group.hi));
            #else
                count = __builtin_popcountll(current_group.lo) + __builtin_popcountll(current_group.hi);
            #endif

            if (count >= config::Rule::kConnectCount) {
                result.num_erased += count;
                result.group_bonus += config::Score::getGroupBonus(count);
                total_erased_color |= current_group;
                color_erased = true;
            }
            remaining &= ~current_group;
        }

        if (color_erased) {
            ++result.num_colors;
            board.setBitboard(c, color_bb & ~total_erased_color);
        }
    }

    if (result.num_erased == 0) {
        return result;
    }

    result.erased = true;

    // 2. Erase adjacent Ojama puyos
    const BitBoard ojama_bb = board.getBitboard(Cell::Ojama);
    if (!ojama_bb.empty()) {
        BitBoard adjacent_mask = total_erased_color.shiftUp();
        adjacent_mask |= total_erased_color.shiftDown();
        adjacent_mask |= total_erased_color.shiftLeft();
        adjacent_mask |= total_erased_color.shiftRight();

        const BitBoard erased_ojama = ojama_bb & adjacent_mask;
        if (!erased_ojama.empty()) {
            board.setBitboard(Cell::Ojama, ojama_bb & ~erased_ojama);
        }
    }

    // 3. Calculate score
    int chain_bonus = config::Score::getChainBonus(chain_number);
    int color_bonus = config::Score::getColorBonus(result.num_colors);
    int total_bonus = chain_bonus + result.group_bonus + color_bonus;
    if (total_bonus < 1) total_bonus = 1;

    result.score = (result.num_erased * 10) * total_bonus;

    return result;
}

Chain::ChainResult Chain::executeChain(Board& board, uint8_t first_color_mask) {
    ChainResult result;
    
    Gravity::execute(board);

    // Step 1
    StepResult step = execute(board, 1, first_color_mask);
    if (step.erased) {
        result.total_score += step.score;
        result.max_chain = 1;
        result.total_erased += step.num_erased;
        Gravity::execute(board);
        
        // Step 2+
        while (true) {
            step = execute(board, result.max_chain + 1, 0x0Fu);
            if (!step.erased) break;
            
            result.total_score += step.score;
            ++result.max_chain;
            result.total_erased += step.num_erased;
            Gravity::execute(board);
        }
    }

    return result;
}

} // namespace puyotan
