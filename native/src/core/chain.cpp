#include <puyotan/core/chain.hpp>
#include <vector>

namespace puyotan {



ErasureData Chain::execute(Board& board, uint8_t color_mask) {
    ErasureData data;
    BitBoard total_erased_mask;

    // 1. Find all erased groups per color
    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) continue;

        const Cell c = static_cast<Cell>(i);
        const BitBoard color_board = board.getBitboard(c);
        if (color_board.empty()) continue;

        // --- Bitwise Connectivity Pruning ('has_2' filter) ---
        // A puyo is part of a 4-group ONLY if it or a neighbor has at least 2 neighbors.
        // Neighbors: U, D, L, R. We compute 'at least 2 of 4' using minimized bitwise ops.
        const BitBoard U = color_board.shiftUp();
        const BitBoard D = color_board.shiftDown();
        const BitBoard L = color_board.shiftLeft();
        const BitBoard R = color_board.shiftRight();

        // Optimized 'at least 2 of 4': (U&D) | (L&R) | ((U|D) & (L|R))
        const BitBoard ud_and = U & D;
        const BitBoard ud_or  = U | D;
        const BitBoard lr_and = L & R;
        const BitBoard lr_or  = L | R;
        BitBoard has_2 = color_board & (ud_and | lr_and | (ud_or & lr_or));

        BitBoard remaining = color_board;
        BitBoard color_erased;

        while (!has_2.empty()) {
            BitBoard seed = has_2.extractLSB();
            if (!(remaining & seed).empty()) {
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
                    data.group_sizes[data.num_groups++] = static_cast<uint8_t>(sz);
                    data.num_erased += sz;
                    color_erased |= group;
                }
                remaining &= ~group;
                has_2 &= ~group; 
            }
        }

        if (!color_erased.empty()) {
            data.erased = true;
            ++data.num_colors;
            total_erased_mask |= color_erased;
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
