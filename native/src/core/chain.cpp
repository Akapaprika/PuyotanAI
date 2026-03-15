#include <puyotan/core/chain.hpp>
#include <algorithm>

namespace puyotan {

BitBoard Chain::findGroups(const BitBoard& color_board, int min_size) {
    BitBoard visited(0, 0);
    BitBoard result(0, 0);
    BitBoard remaining = color_board;

    while (!remaining.empty()) {
        int x = 0, y = 0;
        #ifdef _MSC_VER
            unsigned long index;
            if (_BitScanForward64(&index, remaining.lo)) {
                x = (int)index / 16;
                y = (int)index % 16;
            } else if (_BitScanForward64(&index, remaining.hi)) {
                x = (int)index / 16 + 4;
                y = (int)index % 16;
            }
        #else
            if (remaining.lo != 0) {
                int index = __builtin_ctzll(remaining.lo);
                x = index / 16;
                y = index % 16;
            } else {
                int index = __builtin_ctzll(remaining.hi);
                x = index / 16 + 4;
                y = index % 16;
            }
        #endif

        if (x == -1) {
            break;
        }

        // BFS/Flood-fill to find group
        BitBoard group(0, 0);
        group.set(x, y);
        BitBoard last_group;
        do {
            last_group = group;
            BitBoard expanded = group | group.shiftUp() | group.shiftDown() | group.shiftLeft() | group.shiftRight();
            group = expanded & color_board;
        } while (group != last_group);

        // Count bits in group
        #ifdef _MSC_VER
            int size = (int)(__popcnt64(group.lo) + __popcnt64(group.hi));
        #else
            int size = (int)(__builtin_popcountll(group.lo) + __builtin_popcountll(group.hi));
        #endif

        if (size >= min_size) {
            result |= group;
        }
        remaining &= ~group;
    }
    return result;
}

ErasureData Chain::execute(Board& board, uint8_t color_mask) {
    ErasureData data;
    BitBoard total_erased_mask(0, 0);

    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) {
            continue;
        }

        const Cell c = static_cast<Cell>(i);
        BitBoard color_board = board.getBitboard(c);
        if (color_board.empty()) {
            continue;
        }

        BitBoard groups = findGroups(color_board, config::Rule::kConnectCount);
        if (!groups.empty()) {
            data.erased = true;
            data.num_colors++;
            total_erased_mask |= groups;

            // Extract group sizes for scoring
            BitBoard remaining = groups;
            while (!remaining.empty()) {
                BitBoard single_group(0, 0);
                int x=0, y=0;
                #ifdef _MSC_VER
                    unsigned long idx;
                    if (_BitScanForward64(&idx, remaining.lo)) {
                        x = (int)idx / 16;
                        y = (int)idx % 16;
                    } else if (_BitScanForward64(&idx, remaining.hi)) {
                        x = (int)idx / 16 + 4;
                        y = (int)idx % 16;
                    }
                #else
                    if (remaining.lo != 0) {
                        int index = __builtin_ctzll(remaining.lo);
                        x = index / 16;
                        y = index % 16;
                    } else {
                        int index = __builtin_ctzll(remaining.hi);
                        x = index / 16 + 4;
                        y = index % 16;
                    }
                #endif
                
                single_group.set(x, y);
                BitBoard last_sg;
                do {
                    last_sg = single_group;
                    single_group = (single_group | single_group.shiftUp() | single_group.shiftDown() | single_group.shiftLeft() | single_group.shiftRight()) & groups;
                } while (single_group != last_sg);
                
                #ifdef _MSC_VER
                    int sz = (int)(__popcnt64(single_group.lo) + __popcnt64(single_group.hi));
                #else
                    int sz = (int)(__builtin_popcountll(single_group.lo) + __builtin_popcountll(single_group.hi));
                #endif
                data.group_sizes.push_back(sz);
                data.num_erased += sz;
                remaining &= ~single_group;
            }
            
            board.setBitboard(c, color_board & ~groups, true);
        }
    }

    if (data.erased) {
        // Erase adjacent Ojama
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            const BitBoard adj = total_erased_mask.shiftUp() | total_erased_mask.shiftDown() | 
                                 total_erased_mask.shiftLeft() | total_erased_mask.shiftRight();
            const BitBoard ojama_to_erase = ojama & adj;
            if (!ojama_to_erase.empty()) {
                board.setBitboard(Cell::Ojama, ojama & ~ojama_to_erase, true);
            }
        }
    }

    return data;
}

} // namespace puyotan
