#pragma once

#include <array>
#include <cstdint>
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

    static void scanGroups(const Board& board, ErasureData& erasure_data,
                           uint32_t color_mask = kAllColorsMask) noexcept;

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

            const BitBoard ud_and = U & D;
            const BitBoard ud_or  = U | D;
            const BitBoard lr_and = L & R;
            const BitBoard lr_or  = L | R;

            const BitBoard deg_ge3 =
                color_board & ((ud_and & lr_or) | (lr_and & ud_or));
            const BitBoard deg_ge2 =
                color_board & (ud_and | lr_and | (ud_or & lr_or));
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