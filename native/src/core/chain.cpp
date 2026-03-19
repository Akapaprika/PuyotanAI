#include <puyotan/core/chain.hpp>

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
        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        const BitBoard R = color_board.shiftRightRaw();

        // 'at least 2 of 4': (U&D) | (L&R) | ((U|D) & (L|R))
        const BitBoard ud_and = U & D;
        const BitBoard ud_or  = U | D;
        const BitBoard lr_and = L & R;
        const BitBoard lr_or  = L | R;
        BitBoard has_2 = color_board & (ud_and | lr_and | (ud_or & lr_or));

        BitBoard color_erased;

        while (!has_2.empty()) {
            BitBoard seed = has_2.extractLSB();
            BitBoard group = seed;
            BitBoard prev;
            do {
                prev = group;
                __m128i v = group.m128;
                __m128i neighbors = _mm_or_si128(
                    _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1)),
                    _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2))
                );
                group.m128 = _mm_and_si128(_mm_or_si128(v, neighbors), color_board.m128);
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                data.group_sizes[data.num_groups++] = static_cast<uint8_t>(sz);
                data.num_erased += sz;
                color_erased |= group;
            }
            has_2 &= ~group;
        }

        if (color_erased.popcount() > 0) {
            ++data.num_colors;
            total_erased_mask |= color_erased;
            board.setBitboard(c, color_board & ~color_erased, false);
        }
    }

    // 2. Handle Ojama erasure if any puyos were cleared
    if (data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            BitBoard adj = total_erased_mask;
            adj |= total_erased_mask.shiftUp();
            adj |= total_erased_mask.shiftDown();
            adj |= total_erased_mask.shiftLeft();
            adj |= total_erased_mask.shiftRight();

            const uint64_t oj_erase_lo = ojama.lo & adj.lo;
            const uint64_t oj_erase_hi = ojama.hi & adj.hi;
            if (oj_erase_lo | oj_erase_hi) {
                BitBoard new_ojama;
                new_ojama.lo = ojama.lo & ~oj_erase_lo;
                new_ojama.hi = ojama.hi & ~oj_erase_hi;
                board.setBitboard(Cell::Ojama, new_ojama, false);
                total_erased_mask.lo |= oj_erase_lo;
                total_erased_mask.hi |= oj_erase_hi;
            }
        }

        // Fast O(1) incremental occupancy update
        BitBoard occ = board.getOccupied();
        occ.lo &= ~total_erased_mask.lo;
        occ.hi &= ~total_erased_mask.hi;
        board.updateOccupancy(occ);
    }

    return data;
}

bool Chain::canFire(const Board& board, uint8_t color_mask) {
    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) continue;

        const Cell c = static_cast<Cell>(i);
        const BitBoard color_board = board.getBitboard(c);
        if (color_board.empty()) continue;

        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        const BitBoard R = color_board.shiftRightRaw();

        const BitBoard ud_and = U & D;
        const BitBoard ud_or  = U | D;
        const BitBoard lr_and = L & R;
        const BitBoard lr_or  = L | R;
        BitBoard has_2 = color_board & (ud_and | lr_and | (ud_or & lr_or));

        while (!has_2.empty()) {
            BitBoard seed = has_2.extractLSB();
            BitBoard group = seed;
            BitBoard prev;
            do {
                prev = group;
                __m128i v = group.m128;
                __m128i neighbors = _mm_or_si128(
                    _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1)),
                    _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2))
                );
                group.m128 = _mm_and_si128(_mm_or_si128(v, neighbors), color_board.m128);
                if (group.popcount() >= config::Rule::kConnectCount) return true;
            } while (group != prev);
            has_2 &= ~group;
        }
    }
    return false;
}

} // namespace puyotan
