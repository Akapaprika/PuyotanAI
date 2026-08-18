#include <immintrin.h>
#include <puyotan/core/chain.hpp>

namespace puyotan {

void Chain::scanGroups(const Board& board, ErasureData& erasure_data,
                       uint32_t color_mask) noexcept {
    erasure_data.clear();

    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    uint32_t erased_color_bits = 0;
    uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);

    while (temp_mask) {
        const int i = std::countr_zero(temp_mask);
        const Cell c = static_cast<Cell>(i);

        // 1. 純粋な SIMD (GPR往復なし) でマスク
        const BitBoard color_board =
            _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

        // 2. 2連結チェック (上と左の2シフトのみ・最速のSIMD早期判定)
        const BitBoard U = color_board.shiftUpRaw();
        const BitBoard L = color_board.shiftLeftRaw();
        if ((color_board & (U | L)).empty()) {
            temp_mask &= (temp_mask - 1);
            continue;
        }

        const BitBoard D = color_board.shiftDownRaw();
        const BitBoard R = color_board.shiftRightRaw();

        const BitBoard ud_and = U & D;
        const BitBoard ud_or  = U | D;
        const BitBoard lr_and = L & R;
        const BitBoard lr_or  = L | R;

        const BitBoard deg_ge3 =
            color_board & ((ud_and & lr_or) | (lr_and & ud_or));
        const BitBoard deg_ge2 =
            color_board & (ud_and | lr_and | (ud_or & lr_or));

        // ★ 上と左の 2シフトのみで判定 (下・右シフトと余計なORを完全排除)
        const BitBoard d2_adjacent =
            deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

        BitBoard true_seeds = deg_ge3 | d2_adjacent;
        if (true_seeds.empty()) {
            temp_mask &= (temp_mask - 1);
            continue;
        }

        const __m128i cb_mask = color_board.m128;

        // ★ 全シード並列一括拡張 (複数シードから同時に Flood Fill)
        BitBoard group = true_seeds;
        BitBoard prev;
        do {
            prev = group;
            const __m128i v = group.m128;
            const __m128i ud =
                _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
            const __m128i lr =
                _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
            group.m128 = _mm_and_si128(
                _mm_or_si128(v, _mm_or_si128(ud, lr)), cb_mask);
        } while (group != prev);

        const int sz = group.popcount();

        // ★ 定理: 7個以下なら 100% 単一グループ確定 (4+4=8 のため)
        // 98% 以上のケースで LSB ループをスキップ
        if (sz < 8) {
            erasure_data.group_sizes[erasure_data.num_groups++] =
                static_cast<uint8_t>(sz);
            erasure_data.num_erased += sz;
            erasure_data.total_erased |= group;
            erased_color_bits |= (1u << i);
        } else {
            // レアケース (8個以上 / 同色複数グループ) のみ分離
            BitBoard remaining = true_seeds;
            while (!remaining.empty()) {
                BitBoard g = remaining.extractLSB();
                BitBoard p;
                do {
                    p = g;
                    const __m128i v = g.m128;
                    const __m128i ud =
                        _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                    const __m128i lr =
                        _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                    g.m128 = _mm_and_si128(
                        _mm_or_si128(v, _mm_or_si128(ud, lr)), cb_mask);
                } while (g != p);

                const int single_sz = g.popcount();
                erasure_data.group_sizes[erasure_data.num_groups++] =
                    static_cast<uint8_t>(single_sz);
                erasure_data.num_erased += single_sz;
                erasure_data.total_erased |= g;
                erased_color_bits |= (1u << i);

                remaining.andNot(g);
            }
        }

        temp_mask &= (temp_mask - 1);
    }

    erasure_data.num_colors =
        static_cast<int>(_mm_popcnt_u32(erased_color_bits));

    // おじゃまぷよ消去（完全分岐レス）
    if (erasure_data.num_erased > 0) {
        const BitBoard ojama = board.getBitboard(Cell::Ojama);
        if (!ojama.empty()) {
            const BitBoard& t = erasure_data.total_erased;

            const __m128i raw_up    = _mm_slli_epi64(t.m128, 1);
            const __m128i raw_down  = _mm_srli_epi64(t.m128, 1);
            const __m128i raw_right = _mm_slli_si128(t.m128, 2);
            const __m128i raw_left  = _mm_srli_si128(t.m128, 2);

            const __m128i combined = _mm_or_si128(
                _mm_or_si128(raw_up, raw_down),
                _mm_or_si128(raw_right, raw_left));

            const __m128i boundary_mask = _mm_set_epi64x(
                config::Board::kHiMask, config::Board::kLoMask);
            const __m128i adj = _mm_and_si128(combined, boundary_mask);
            const __m128i oj_erased = _mm_and_si128(ojama.m128, adj);

            erasure_data.total_erased.m128 =
                _mm_or_si128(erasure_data.total_erased.m128, oj_erased);
        }
    }
}

void Chain::applyErasure(Board& board, const ErasureData& data) noexcept {
    for (int i = 0; i < config::Board::kNumColors; ++i) {
        board.boards_[i].andNot(data.total_erased);
    }
    board.occupancy_.andNot(data.total_erased);
}

ErasureData Chain::execute(Board& board, uint32_t color_mask) noexcept {
    ErasureData data;
    execute(board, data, color_mask);
    return data;
}

void Chain::execute(Board& board, ErasureData& data,
                    uint32_t color_mask) noexcept {
    scanGroups(board, data, color_mask);
    if (data.num_erased > 0) {
        applyErasure(board, data);
    }
}

} // namespace puyotan