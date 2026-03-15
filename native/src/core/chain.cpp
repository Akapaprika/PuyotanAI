#include <puyotan/core/chain.hpp>

namespace puyotan {

std::vector<BitBoard> Chain::findGroups(const BitBoard& color_board, int min_size) {
    std::vector<BitBoard> result;
    BitBoard remaining = color_board;

    while (!remaining.empty()) {
        // Pick a puyo as a seed using the LSB trick (branchless candidate extraction)
        BitBoard seed = remaining.extractLSB();
        
        // Flood-fill to find group
        BitBoard group = seed;
        BitBoard last_group;
        do {
            last_group = group;
            // Expand group in all 4 directions efficiently
            group |= group.shiftUp();
            group |= group.shiftDown();
            group |= group.shiftLeft();
            group |= group.shiftRight();
            group &= color_board;
        } while (group != last_group);

        // Record group if it meets the size requirement
        if (group.popcount() >= min_size) {
            result.push_back(group);
        }
        
        // Remove the entire processed group (regardless of size) from the search board
        remaining &= ~group;
    }
    return result;
}

ErasureData Chain::execute(Board& board, uint8_t color_mask) {
    ErasureData data;
    BitBoard total_erased_mask;

    // 1. Find all erased groups per color
    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) {
            continue;
        }

        const Cell c = static_cast<Cell>(i);
        BitBoard color_board = board.getBitboard(c);
        if (color_board.empty()) {
            continue;
        }

        // Single pass: find groups and collect metadata
        std::vector<BitBoard> groups = findGroups(color_board, config::Rule::kConnectCount);
        if (!groups.empty()) {
            data.erased = true;
            data.num_colors++;
            
            for (const auto& group : groups) {
                int sz = group.popcount();
                data.group_sizes.push_back(sz);
                data.num_erased += sz;
                total_erased_mask |= group;
            }
            
            // Update the board for this color
            board.setBitboard(c, color_board & ~total_erased_mask, true);
        }
    }

    // 2. Handle Ojama erasure if any puyos were cleared
    if (data.erased) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            // Expansion for adjacent Ojama (O(1) SIMD)
            BitBoard adj = total_erased_mask;
            adj |= total_erased_mask.shiftUp();
            adj |= total_erased_mask.shiftDown();
            adj |= total_erased_mask.shiftLeft();
            adj |= total_erased_mask.shiftRight();
            
            const BitBoard ojama_to_erase = ojama & adj;
            if (!ojama_to_erase.empty()) {
                board.setBitboard(Cell::Ojama, ojama & ~ojama_to_erase, true);
            }
        }
    }

    return data;
}

} // namespace puyotan
