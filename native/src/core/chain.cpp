#include <puyotan/core/chain.hpp>

namespace puyotan {

ErasureData Chain::execute(Board& board, uint32_t color_mask) noexcept {
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
        const __m128i cb_mask = color_board.m128; 
        while (!has_2.empty()) {
            BitBoard group = has_2.extractLSB();
            BitBoard prev;
            do {
                prev = group;
                __m128i v = group.m128;
                // Horizontal neighbors (left/right)
                __m128i lr = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                // Vertical neighbors (up/down: 16 bits = 2 bytes)
                __m128i ud = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                // Union of self and all neighbors
                __m128i merged = _mm_or_si128(v, _mm_or_si128(lr, ud));
                group.m128 = _mm_and_si128(merged, cb_mask);
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                data.group_sizes[data.num_groups++] = static_cast<uint8_t>(sz);
                data.num_erased += sz;
                color_erased |= group;
            }
            has_2.andNot(group);
        }

        if (!color_erased.empty()) {
            ++data.num_colors;
            total_erased_mask |= color_erased;
            board.setBitboard(c, BitBoard::andNot(color_board, color_erased));
        }
    }

    // 2. Handle Ojama erasure if any puyos were cleared
    if (data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            BitBoard adj = total_erased_mask | 
                           total_erased_mask.shiftUp() | 
                           total_erased_mask.shiftDown() | 
                           total_erased_mask.shiftLeft() | 
                           total_erased_mask.shiftRight();

            BitBoard oj_erased = ojama & adj;
            if (!oj_erased.empty()) {
                BitBoard new_ojama = ojama;
                new_ojama.andNot(oj_erased);
                board.setBitboard(Cell::Ojama, new_ojama);
                total_erased_mask |= oj_erased;
            }
        }

        // Fast O(1) incremental occupancy update
        BitBoard occ = board.getOccupied();
        occ.andNot(total_erased_mask);
        board.updateOccupancy(occ);
    }

    return data;
}

bool Chain::canFire(const Board& board, uint32_t color_mask) noexcept {
    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) continue;

        const Cell c = static_cast<Cell>(i);
        const BitBoard color_board = board.getBitboard(c);
        if (color_board.empty()) continue;
        const __m128i cb_mask = color_board.m128;

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
            BitBoard group = has_2.extractLSB();
            BitBoard prev;
            do {
                prev = group;
                __m128i v = group.m128;
                // Horizontal neighbors (left/right)
                __m128i lr = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                // Vertical neighbors (up/down: 16 bits = 2 bytes)
                __m128i ud = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                // Union of self and all neighbors
                __m128i merged = _mm_or_si128(v, _mm_or_si128(lr, ud));
                group.m128 = _mm_and_si128(merged, cb_mask);
                if (group.popcount() >= config::Rule::kConnectCount) return true;
            } while (group != prev);
            has_2.andNot(group);
        }
    }
    return false;
}

} // namespace puyotan
