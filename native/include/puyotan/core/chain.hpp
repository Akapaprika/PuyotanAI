#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/core/board.hpp>

namespace puyotan {

struct ErasureData {
    BitBoard total_erased;
    std::array<uint8_t, config::Rule::kMaxErasureGroups> group_sizes;
    int num_erased = 0;
    int num_colors = 0;
    int num_groups = 0;

    __forceinline void clear() noexcept {
        total_erased = BitBoard();
        num_erased = 0;
        num_colors = 0;
        num_groups = 0;
    }
};

class Chain {
  public:
    static constexpr uint32_t kAllColorsMask =
        (1u << config::Rule::kColors) - 1u;

    static ErasureData execute(Board& board,
                               uint32_t color_mask = kAllColorsMask) noexcept;

    static void execute(Board& board, ErasureData& data,
                        uint32_t color_mask = kAllColorsMask) noexcept;

    static __forceinline void scanGroups(const Board& board, ErasureData& erasure_data,
                                        uint32_t color_mask = kAllColorsMask) noexcept {
        erasure_data.clear();

        const __m128i chainable_mask = _mm_set_epi64x(
            config::Board::kChainableHiMask, config::Board::kChainableLoMask);

        uint32_t erased_color_bits = 0;
        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);

        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const BitBoard color_board =
                _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            // 2連結チェック (上・左の2シフトのみ)
            const BitBoard U = color_board.shiftUpRaw();
            const BitBoard L = color_board.shiftLeftRaw();
            if ((color_board & (U | L)).empty()) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const BitBoard D = color_board.shiftDownRaw();
            const BitBoard R = color_board.shiftRightRaw();

            // ★ ブール代数の因数分解による最速シード抽出 (8命令 → 6命令)
            const BitBoard X = (U & D) | (L & R); // 縦または横の3連中心
            const BitBoard Y = (U | D) & (L | R); // 縦かつ横の直交隣接

            const BitBoard deg_ge3 = color_board & (X & Y); // T字・十字
            const BitBoard deg_ge2 = color_board & (X | Y); // 2個以上の隣接

            const BitBoard d2_adjacent =
                deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

            BitBoard true_seeds = deg_ge3 | d2_adjacent;
            if (true_seeds.empty()) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const __m128i cb_mask = color_board.m128;

            // 全シード並列一括拡張
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

            // ★ 1. 共通処理を一括確定 (if/else の重複を完全撤廃)
            erasure_data.total_erased |= group;
            erasure_data.num_erased   += sz;
            erased_color_bits         |= (1u << i);

            // ★ 2. グループサイズの記録
            if (sz < 8) {
                erasure_data.group_sizes[erasure_data.num_groups++] = static_cast<uint8_t>(sz);
            } else {
                // 探索対象を group 自体に絞って各成分のサイズを記録
                BitBoard rem = group;
                while (!rem.empty()) {
                    BitBoard g = rem.extractLSB();
                    BitBoard p;
                    do {
                        p = g;
                        const __m128i v = g.m128;
                        const __m128i ud = _mm_or_si128(_mm_slli_epi64(v, 1), _mm_srli_epi64(v, 1));
                        const __m128i lr = _mm_or_si128(_mm_slli_si128(v, 2), _mm_srli_si128(v, 2));
                        g.m128 = _mm_and_si128(_mm_or_si128(v, _mm_or_si128(ud, lr)), group.m128);
                    } while (g != p);

                    const int single_sz = g.popcount();
                    erasure_data.group_sizes[erasure_data.num_groups++] = static_cast<uint8_t>(single_sz);
                    rem.andNot(g);
                }
            }

            temp_mask &= (temp_mask - 1);
        }

        erasure_data.num_colors =
            static_cast<int>(_mm_popcnt_u32(erased_color_bits));

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

    static void applyErasure(Board& board, const ErasureData& data) noexcept;

    [[nodiscard]] static __forceinline bool canFire(const Board& board,
                                                uint32_t color_mask = kAllColorsMask) noexcept {
        const __m128i chainable_mask = _mm_set_epi64x(
            config::Board::kChainableHiMask, config::Board::kChainableLoMask);

        uint32_t temp_mask = color_mask & ((1u << config::Rule::kColors) - 1u);
        while (temp_mask) {
            const int i = std::countr_zero(temp_mask);
            const Cell c = static_cast<Cell>(i);

            const BitBoard color_board =
                _mm_and_si128(board.getBitboard(c).m128, chainable_mask);

            const BitBoard U = color_board.shiftUpRaw();
            const BitBoard L = color_board.shiftLeftRaw();
            if ((color_board & (U | L)).empty()) {
                temp_mask &= (temp_mask - 1);
                continue;
            }

            const BitBoard D = color_board.shiftDownRaw();
            const BitBoard R = color_board.shiftRightRaw();

            // ★ ブール束因数分解: 共通項 X と Y を計算
            const BitBoard X = (U & D) | (L & R); // 3連の中心 (縦または横)
            const BitBoard Y = (U | D) & (L | R); // L字の角 (縦かつ横)

            // deg_ge3 = X & Y, deg_ge2 = X | Y (わずか 2本の命令で両方完成！)
            const BitBoard deg_ge3 = color_board & (X & Y);
            const BitBoard deg_ge2 = color_board & (X | Y);

            const BitBoard d2_adjacent =
                deg_ge2 & (deg_ge2.shiftUpRaw() | deg_ge2.shiftLeftRaw());

            if (!(deg_ge3 | d2_adjacent).empty()) {
                return true;
            }

            temp_mask &= (temp_mask - 1);
        }
        return false;
    }
};

} // namespace puyotan