#include <puyotan/core/chain.hpp>

namespace puyotan {

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
        // 1が立っている最も下位のビット位置（色番号）を取得（TZCNT命令: 1サイクル）
        const int i = std::countr_zero(temp_mask);
        const Cell c = static_cast<Cell>(i);

        const BitBoard& bb = board.getBitboard(c);
        const uint64_t lo_masked = bb.lo & config::Board::kChainableLoMask;
        const uint64_t hi_masked = bb.hi & config::Board::kChainableHiMask;

        const BitBoard color_board(_mm_set_epi64x(
            static_cast<int64_t>(hi_masked), static_cast<int64_t>(lo_masked)));

        // ★ 1. 4方向の生の高速シフト演算
        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        const BitBoard R = color_board.shiftRightRaw();

        // ★ 2. 隣接するぷよの集合（adj）を一括計算
        BitBoard adj = color_board & (U | D | L | R);

        // ★ 3. 【超絶高速化】隣接ぷよが合計4個未満なら、絶対に4連結になり得ないので1サイクルで即スキップ！
        if (adj.popcount() < config::Rule::kConnectCount) {
            temp_mask &= (temp_mask - 1);
            continue;
        }

        const __m128i cb_mask = color_board.m128;

        // ★ 4. adj 自体をシードとして使用（従来必要だった8個のSIMD論理演算を完全削除）
        while (!adj.empty()) {
            BitBoard group = adj.extractLSB();
            BitBoard prev;

            // 連結グループの超高速膨張（Flood Fill）
            do {
                prev = group;
                const __m128i v = group.m128;
                // 上下（1ビットシフト）と 左右（16ビット＝2バイトシフト）の膨張
                const __m128i ud = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                const __m128i lr = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));

                group.m128 = _mm_and_si128(
                    _mm_or_si128(v, _mm_or_si128(ud, lr)), cb_mask);
            } while (group != prev);

            const int sz = group.popcount();
            if (sz >= config::Rule::kConnectCount) {
                erasure_data.group_sizes[erasure_data.num_groups++] =
                    static_cast<uint8_t>(sz);
                erasure_data.num_erased += sz;

                erasure_data.total_erased |= group;
                erased_color_bits |= (1u << i);
            }

            // 調査済みグループの全ぷよを adj から一括削除して次のグループへ
            adj.andNot(group);
        }

        temp_mask &= (temp_mask - 1);
    }

    erasure_data.num_colors =
        static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

    // ★ 5. おじゃまぷよの消去処理（変更なし・高速SIMDのまま）
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

            const __m128i boundary_mask = _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask);
            const BitBoard adj_ojama = _mm_and_si128(combined, boundary_mask);
            const BitBoard oj_erased = ojama & adj_ojama;

            if (!oj_erased.empty()) {
                erasure_data.total_erased |= oj_erased;
            }
        }
    }
}

void Chain::applyErasure(Board& board, const ErasureData& data) noexcept {
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