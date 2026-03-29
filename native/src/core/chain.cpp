#include <puyotan/core/chain.hpp>

namespace puyotan {

// -----------------------------------------------------------------------
// Internal BFS kernel shared by both findGroups and the legacy canFire path.
// Fills data.erased_per_color[i], data.total_erased, group_sizes, and counts.
// Does NOT touch the Board.
// -----------------------------------------------------------------------
static void scanGroups(const Board& board, uint32_t color_mask, ErasureData& data) noexcept {
    uint32_t erased_color_bits = 0; // Bit i is set if color i was erased

    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1)) continue;
        const Cell c = static_cast<Cell>(i);
        
        // Mask ghost puyos (y >= 12) so they don't participate in connectivity or erasure.
        static const __m128i kGhostMask = _mm_set_epi64x(config::Board::kChainableHiMask, config::Board::kChainableLoMask);
        const BitBoard color_board(_mm_and_si128(board.getBitboard(c).m128, kGhostMask));
        if (color_board.empty()) continue;

        // Bitwise Connectivity Pruning ('has_2' filter):
        // Only puyos with >= 2 neighbors of the same color can be part of a 4+ group.
        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        const BitBoard R = color_board.shiftRightRaw();

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
                // SIMD BFS: Expand seed 'group' by 1 step in all 4 directions.
                __m128i lr = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                __m128i ud = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                group.m128 = _mm_and_si128(_mm_or_si128(v, _mm_or_si128(lr, ud)), cb_mask);
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                data.group_sizes[data.num_groups++] = static_cast<uint8_t>(sz);
                data.num_erased += sz;
                color_erased |= group;          // Accumulate for this color
                data.total_erased |= group;     // Accumulate for overall erasure
                erased_color_bits |= (1u << i); // Record that this color had an erasure
            }
            has_2.andNot(group);
        }

        // Store result for this color (No branch: safe even if color_erased is empty)
        data.erased_per_color[i] = color_erased;
    }

    // Convert bitmask to color count in a single instruction
    data.num_colors = static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

    // Ojama adjacency: erased if adjacent to any color erasure
    if (data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            const BitBoard& t = data.total_erased;
            BitBoard adj = t | t.shiftUp() | t.shiftDown() | t.shiftLeft() | t.shiftRight();
            BitBoard oj_erased = ojama & adj;
            if (!oj_erased.empty()) {
                data.erased_per_color[static_cast<int>(Cell::Ojama)] = oj_erased;
                data.total_erased |= oj_erased;
            }
        }
    }
}

// -----------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------

ErasureData Chain::findGroups(const Board& board, uint32_t color_mask) noexcept {
    ErasureData data;
    scanGroups(board, color_mask, data);
    return data;
}

void Chain::applyErasure(Board& board, const ErasureData& data) noexcept {
    // Apply per-color erasures using cached BitBoard masks (no re-scan)
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        const BitBoard& erased = data.erased_per_color[i];
        if (erased.empty()) continue;
        const Cell c = static_cast<Cell>(i);
        board.setBitboard(c, BitBoard::andNot(board.getBitboard(c), erased));
    }
    // Fast O(1) incremental occupancy update
    BitBoard occ = board.getOccupied();
    occ.andNot(data.total_erased);
    board.updateOccupancy(occ);
}

ErasureData Chain::execute(Board& board, uint32_t color_mask) noexcept {
    ErasureData data = findGroups(board, color_mask);
    if (data.num_erased > 0) {
        applyErasure(board, data);
    }
    return data;
}

bool Chain::canFire(const Board& board, uint32_t color_mask) noexcept {
    return findGroups(board, color_mask).num_erased > 0;
}

} // namespace puyotan
