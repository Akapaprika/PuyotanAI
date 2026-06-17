#include <puyotan/core/chain.hpp>

namespace puyotan {
namespace {
// kGhostMask: masks out the ghost row (y >= 12) so those puyos don't
// participate in connectivity or erasure.
// Placed at file scope to avoid MSVC's per-call thread-safe static
// initialization (hidden mutex) that would fire on every scanGroups() call.
static const __m128i kGhostMask =
    _mm_set_epi64x(static_cast<int64_t>(config::Board::kChainableHiMask),
                   static_cast<int64_t>(config::Board::kChainableLoMask));
} // anonymous namespace

// -----------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------

void Chain::scanGroups(const Board& board, ErasureData& erasure_data,
                       uint32_t color_mask) noexcept {
    erasure_data.num_erased = 0;
    erasure_data.num_colors = 0;
    erasure_data.num_groups = 0;
    erasure_data.total_erased = BitBoard();

    uint32_t erased_color_bits = 0;

    for (int i = 0; i < config::Rule::kColors; ++i) {
        if (!((color_mask >> i) & 1))
            continue;
        const Cell c = static_cast<Cell>(i);

        const BitBoard& bb = board.getBitboard(c);
        const uint64_t lo_masked = bb.lo & config::Board::kChainableLoMask;
        const uint64_t hi_masked = bb.hi & config::Board::kChainableHiMask;

        const int pop = static_cast<int>(std::popcount(lo_masked) +
                                         std::popcount(hi_masked));

        if (pop < config::Rule::kConnectCount)
            continue;

        const BitBoard color_board(_mm_and_si128(bb.m128, kGhostMask));

        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        const BitBoard R = color_board.shiftRightRaw();

        const BitBoard ud_and = U & D;
        const BitBoard ud_or = U | D;
        const BitBoard lr_and = L & R;
        const BitBoard lr_or = L | R;
        BitBoard has_2 = color_board & (ud_and | lr_and | (ud_or & lr_or));

        const __m128i cb_mask = color_board.m128;
        while (!has_2.empty()) {
            BitBoard group = has_2.extractLSB();
            BitBoard prev;
            do {
                prev = group;
                __m128i v = group.m128;
                __m128i lr =
                    _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                __m128i ud =
                    _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                group.m128 = _mm_and_si128(
                    _mm_or_si128(v, _mm_or_si128(lr, ud)), cb_mask);
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                erasure_data.group_sizes[erasure_data.num_groups++] =
                    static_cast<uint8_t>(sz);
                erasure_data.num_erased += sz;

                // color_erased を介さず、直接全体の total_erased
                // にのみ OR 結合する
                erasure_data.total_erased |= group;
                erased_color_bits |= (1u << i);
            }
            has_2.andNot(group);
        }
    }

    erasure_data.num_colors =
        static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

    if (erasure_data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            const BitBoard& t = erasure_data.total_erased;
            BitBoard adj = t | t.shiftUp() | t.shiftDown() | t.shiftLeft() |
                           t.shiftRight();
            BitBoard oj_erased = ojama & adj;
            if (!oj_erased.empty()) {
                // こちらも全体の total_erased にのみ結合する
                erasure_data.total_erased |= oj_erased;
            }
        }
    }
}

void Chain::applyErasure(Board& board, const ErasureData& data) noexcept {
    // 色別のマスクではなく、1つの結合された消去マスク data.total_erased
    // を使って、 全色一律で andNot します。
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        board.boards_[i].andNot(data.total_erased);
    }

    BitBoard occ = board.getOccupied();
    occ.andNot(data.total_erased);
    board.updateOccupancy(occ);
}

ErasureData Chain::execute(Board& board, uint32_t color_mask) noexcept {
    ErasureData data;
    scanGroups(board, data, color_mask);
    if (data.num_erased > 0) {
        applyErasure(board, data);
    }
    return data;
}

bool Chain::canFire(const Board& board, uint32_t color_mask) noexcept {
    ErasureData data;
    scanGroups(board, data, color_mask);
    return data.num_erased > 0;
}
} // namespace puyotan
