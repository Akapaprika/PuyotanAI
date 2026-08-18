#include <immintrin.h>
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

    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    uint32_t erased_color_bits = 0;
    uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);

    while (temp_mask) {
        const int i = std::countr_zero(temp_mask);
        const Cell c = static_cast<Cell>(i);

        const BitBoard color_board =
            _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

        // 1. 2連結の超高速事前チェック
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

        // ★ 次数3以上のマスク: (U&D and (L|R)) | (L&R and (U|D))
        const BitBoard deg_ge3 =
            color_board & ((ud_and & lr_or) | (lr_and & ud_or));

        // ★ 次数2以上のマスク:
        const BitBoard deg_ge2 =
            color_board & (ud_and | lr_and | (ud_or & lr_or));

        // ★ 次数2同士が隣接している箇所のマスク:
        const BitBoard d2_ud = deg_ge2.shiftUpRaw() | deg_ge2.shiftDownRaw();
        const BitBoard d2_lr = deg_ge2.shiftLeftRaw() | deg_ge2.shiftRightRaw();
        const BitBoard d2_adjacent = deg_ge2 & (d2_ud | d2_lr);

        // ★ 定理: 4連結に属するぷよのみを 100% 抽出した真のシード (3個ぷよは完全に0になる)
        BitBoard true_seeds = deg_ge3 | d2_adjacent;

        const __m128i cb_mask = color_board.m128;
        while (!true_seeds.empty()) {
            BitBoard group = true_seeds.extractLSB();
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

            // このシードから探索されたグループは、定理より確実に 4連結以上！
            const int sz = group.popcount();
            erasure_data.group_sizes[erasure_data.num_groups++] =
                static_cast<uint8_t>(sz);
            erasure_data.num_erased += sz;
            erasure_data.total_erased |= group;
            erased_color_bits |= (1u << i);

            true_seeds.andNot(group);
        }
        temp_mask &= (temp_mask - 1);
    }

    erasure_data.num_colors =
        static_cast<uint8_t>(_mm_popcnt_u32(erased_color_bits));

    // おじゃまぷよ消去
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
    scanGroups(board, data, color_mask);
    if (data.num_erased > 0) {
        applyErasure(board, data);
    }
    return data;
}

/**
 * @brief 【完全ループレス・O(1) 数学判定版】 canFire
 * グラフ理論の定理により、ループ・Flood Fill・Popcount を一切使わずに
 * わずか十数命令のビット演算だけで 4連結の有無を判定
 */
bool Chain::canFire(const Board& board, uint32_t color_mask) noexcept {
    const __m128i chainable_mask = _mm_set_epi64x(
        config::Board::kChainableHiMask, config::Board::kChainableLoMask);

    uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);
    while (temp_mask) {
        const int i = std::countr_zero(temp_mask);
        const Cell c = static_cast<Cell>(i);

        const BitBoard color_board =
            _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

        // 1. 2連結チェック
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

        // ★ 次数3以上の存在判定
        const BitBoard deg_ge3 =
            color_board & ((ud_and & lr_or) | (lr_and & ud_or));

        // ★ 次数2以上の隣接判定 (対称性により 上と左 のシフトだけで判定可能)
        const BitBoard deg_ge2 =
            color_board & (ud_and | lr_and | (ud_or & lr_or));
        const BitBoard d2_adjacent =
            deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

        // ★ 定理: deg_ge3 または d2_adjacent が非ゼロなら、100% 確実に4連結が存在する！
        if (!(deg_ge3 | d2_adjacent).empty()) {
            return true; // ループなしで即座に判定完了！
        }

        temp_mask &= (temp_mask - 1);
    }
    return false;
}

} // namespace puyotan