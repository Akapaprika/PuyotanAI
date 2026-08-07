#include <puyotan/core/chain.hpp>

namespace puyotan {
namespace {
static const alignas(16) uint64_t kBoundaryMaskData[2] = {
    config::Board::kLoMask, config::Board::kHiMask
};
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
    uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);
    while (temp_mask) {
        // 1が立っている最も下位のビット位置（色番号）を一瞬で取得（TZCNT命令:
        // 1サイクル）
        const int i = std::countr_zero(temp_mask);
        const Cell c = static_cast<Cell>(i);

        const BitBoard& bb = board.getBitboard(c);
        const uint64_t lo_masked = bb.lo & config::Board::kChainableLoMask;
        const uint64_t hi_masked = bb.hi & config::Board::kChainableHiMask;

        const BitBoard color_board(_mm_set_epi64x(
            static_cast<int64_t>(hi_masked), static_cast<int64_t>(lo_masked)));

        // ★ 1. 接続性の超高速チェック（上と左の2シフトのみ）
        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard L = color_board.shiftLeftRaw();

        if ((color_board & (U | L)).empty()) {
            temp_mask &= (temp_mask - 1);
            continue; // 隣接するペアが1組もないため、同値性を保ったまま安全にスキップ
        }

        // ★ 2. スキップを免れた（実際に隣接がある）場合のみ、残りの D と R
        // を計算する
        const BitBoard D = color_board.shiftDownRaw();
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

                erasure_data.total_erased |= group;
                erased_color_bits |= (1u << i);
            }
            has_2.andNot(group);
        }
        temp_mask &= (temp_mask - 1);
    }

    erasure_data.num_colors =
        static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

    if (erasure_data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            const BitBoard& t = erasure_data.total_erased;
        
            const __m128i raw_up    = _mm_slli_epi64(t.m128, 1);
            const __m128i raw_down  = _mm_srli_epi64(t.m128, 1);
            const __m128i raw_right = _mm_slli_si128(t.m128, 2);
            const __m128i raw_left  = _mm_srli_si128(t.m128, 2);
        
            const __m128i combined = _mm_or_si128(t.m128, 
                _mm_or_si128(_mm_or_si128(raw_up, raw_down),
                            _mm_or_si128(raw_right, raw_left)));
        
            // 動的組み立てを排除し、16バイトアライメント領域から 1命令 (movaps) で直接ロード
            const __m128i boundary_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(kBoundaryMaskData));
            const BitBoard adj = _mm_and_si128(combined, boundary_mask);
            BitBoard oj_erased = ojama & adj;
            if (!oj_erased.empty()) {
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
