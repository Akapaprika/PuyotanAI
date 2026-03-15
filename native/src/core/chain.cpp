#include <puyotan/core/chain.hpp>
#include <vector>

namespace puyotan {

std::vector<BitBoard> Chain::findGroups(const BitBoard& color_board, int min_size) {
    std::vector<BitBoard> result;
    BitBoard remaining = color_board;

    while (!remaining.empty()) {
        BitBoard seed = remaining.extractLSB();

        BitBoard group = seed;
        BitBoard last_group;
        do {
            last_group = group;
            group |= group.shiftUp();
            group |= group.shiftDown();
            group |= group.shiftLeft();
            group |= group.shiftRight();
            group &= color_board;
        } while (group != last_group);

        if (group.popcount() >= min_size) {
            result.push_back(group);
        }

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

        // --- Inline group flood-fill (avoids returning a std::vector<BitBoard>) ---
        BitBoard remaining = color_board;
        BitBoard color_erased;

        while (!remaining.empty()) {
            BitBoard seed = remaining.extractLSB();

            // Flood-fill: all 4 shifts computed from the SAME group snapshot.
            // Out-of-order CPU can issue them in parallel — dependency depth: 4 → 2.
            BitBoard group = seed;
            BitBoard prev;
            do {
                prev = group;
                const BitBoard expand = (group.shiftUp() | group.shiftDown()) |
                                        (group.shiftLeft() | group.shiftRight());
                group = (group | expand) & color_board;
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                // Max possible groups << 24: bounds check is redundant, removed for speed.
                data.group_sizes[data.num_groups++] = static_cast<uint8_t>(sz);
                data.num_erased += sz;
                color_erased |= group;
            }

            remaining &= ~group;
        }

        if (!color_erased.empty()) {
            data.erased = true;
            data.num_colors++;
            total_erased_mask |= color_erased;
            // Defer occupancy update — done in one pass after all colors are processed
            board.setBitboard(c, color_board & ~color_erased, false);
        }
    }

    // 2. Handle Ojama erasure if any puyos were cleared
    if (data.erased) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            BitBoard adj = total_erased_mask;
            adj |= total_erased_mask.shiftUp();
            adj |= total_erased_mask.shiftDown();
            adj |= total_erased_mask.shiftLeft();
            adj |= total_erased_mask.shiftRight();

            const BitBoard ojama_to_erase = ojama & adj;
            if (!ojama_to_erase.empty()) {
                board.setBitboard(Cell::Ojama, ojama & ~ojama_to_erase, false);
            }
        }

        // Recompute occupancy ONCE after all boards have been updated
        BitBoard occ = board.getBitboard(Cell::Red);
        for (int i = 1; i < config::Board::kNumColors; ++i) {
            occ |= board.getBitboard(static_cast<Cell>(i));
        }
        board.updateOccupancy(occ);
    }

    return data;
}

} // namespace puyotan
