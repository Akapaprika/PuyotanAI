#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>

#include <immintrin.h>

namespace puyotan {

struct alignas(16) BitBoard {
    union {
        __m128i m128;
        struct {
            uint64_t lo;
            uint64_t hi;
        };
        uint16_t cols[8];
    };

    BitBoard() noexcept : m128(_mm_setzero_si128()) {}
    constexpr BitBoard(uint64_t l, uint64_t h) noexcept : lo(l), hi(h) {}
    BitBoard(__m128i m) noexcept : m128(m) {}

    [[nodiscard]] __forceinline bool operator==(const BitBoard& o) const noexcept {
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) != 0;
    }
    [[nodiscard]] __forceinline bool operator!=(const BitBoard& o) const noexcept {
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) == 0;
    }
    [[nodiscard]] __forceinline BitBoard operator&(const BitBoard& o) const noexcept {
        return _mm_and_si128(m128, o.m128);
    }
    [[nodiscard]] __forceinline BitBoard operator|(const BitBoard& o) const noexcept {
        return _mm_or_si128(m128, o.m128);
    }
    [[nodiscard]] __forceinline BitBoard operator~() const noexcept {
        return _mm_xor_si128(m128, _mm_set1_epi32(-1));
    }
    __forceinline BitBoard& operator&=(const BitBoard& o) noexcept {
        m128 = _mm_and_si128(m128, o.m128);
        return *this;
    }
    __forceinline BitBoard& operator|=(const BitBoard& o) noexcept {
        m128 = _mm_or_si128(m128, o.m128);
        return *this;
    }
    __forceinline BitBoard& andNot(const BitBoard& o) noexcept {
        m128 = _mm_andnot_si128(o.m128, m128);
        return *this;
    }
    [[nodiscard]] static __forceinline BitBoard andNot(const BitBoard& a, const BitBoard& b) noexcept {
        return _mm_andnot_si128(b.m128, a.m128);
    }

    [[nodiscard]] __forceinline bool empty() const noexcept {
        return _mm_testz_si128(m128, m128) != 0;
    }

    [[nodiscard]] __forceinline bool get(int x, int y) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        return (cols[x] >> y) & 1;
    }
    __forceinline void set(int x, int y) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        cols[x] |= static_cast<uint16_t>(1U << y);
    }
    __forceinline void clear(int x, int y) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        cols[x] &= static_cast<uint16_t>(~(1U << y));
    }

    [[nodiscard]] __forceinline int popcount() const noexcept {
        return static_cast<int>(std::popcount(lo) + std::popcount(hi));
    }

    [[nodiscard]] __forceinline BitBoard extractLSB() const noexcept {
        const __m128i zero = _mm_setzero_si128();
        const __m128i neg  = _mm_sub_epi64(zero, m128);
        const __m128i lsb  = _mm_and_si128(m128, neg);
        const __m128i cmp  = _mm_cmpeq_epi64(m128, zero);
        const __m128i mask = _mm_unpacklo_epi64(_mm_set1_epi32(-1), cmp);
        return _mm_and_si128(lsb, mask);
    }

    [[nodiscard]] __forceinline BitBoard shiftUpRaw() const noexcept {
        return _mm_slli_epi64(m128, 1);
    }
    [[nodiscard]] __forceinline BitBoard shiftDownRaw() const noexcept {
        return _mm_srli_epi64(m128, 1);
    }
    [[nodiscard]] __forceinline BitBoard shiftRightRaw() const noexcept {
        return _mm_slli_si128(m128, 2);
    }
    [[nodiscard]] __forceinline BitBoard shiftLeftRaw() const noexcept {
        return _mm_srli_si128(m128, 2);
    }
};

class Board {
  public:
    struct ActivePuyo {
        int x;
        int y;
        Cell color;
    };

    Board() noexcept = default;

    // Python / GUI / テスト用（実体は board.cpp）
    [[nodiscard]] std::vector<ActivePuyo> getActivePuyos() const noexcept;
    [[nodiscard]] Cell get(int x, int y) const noexcept;
    void set(int x, int y, Cell color) noexcept;
    void clear(int x, int y) noexcept;

    // C++ ホットパス用（インライン）
    [[nodiscard]] __forceinline int getColumnHeight(int x) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        return static_cast<int>(_mm_popcnt_u32(occupancy_.cols[x]));
    }

    __forceinline void dropNewPiece(int x, int y, Cell color) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(toIndex(color) >= 0 && toIndex(color) < config::Board::kNumColors);
        const uint16_t bit = static_cast<uint16_t>((1U << y) & config::Board::kVisibleColMask);
        boards_[toIndex(color)].cols[x] |= bit;
        occupancy_.cols[x] |= bit;
    }

    __forceinline void dropPiecePair(int col, Rotation r, Cell color_axis, Cell color_sub, int& out_h_axis, int& out_h_sub) noexcept {
        const int r_idx = static_cast<int>(r);
        const int x_axis = col;
        const int x_sub = col + kSubDx[r_idx];

        const uint32_t lane_axis = occupancy_.cols[x_axis];
        const uint32_t lane_sub  = occupancy_.cols[x_sub];

        // 各列の 1番目の空きマス（着地ビット）を取得
        const uint32_t bit1_axis = _blsi_u32(~lane_axis);
        const uint32_t bit1_sub  = _blsi_u32(~lane_sub);

        out_h_axis = std::countr_zero(bit1_axis);
        out_h_sub  = std::countr_zero(bit1_sub);

        // ★ 上に乗る側だけシフト量を 1 (1段上)、下側や横置きは 0 (空きマスそのまま)
        const int use_2nd_axis = static_cast<int>(r == Rotation::Down);
        const int use_2nd_sub  = static_cast<int>(r == Rotation::Up);

        // シフトを適用し、表示領域（0〜12行）マスクをかける
        const uint16_t bit_axis = static_cast<uint16_t>((bit1_axis << use_2nd_axis) & config::Board::kVisibleColMask);
        const uint16_t bit_sub  = static_cast<uint16_t>((bit1_sub  << use_2nd_sub)  & config::Board::kVisibleColMask);

        boards_[static_cast<int>(color_axis)].cols[x_axis] |= bit_axis;
        boards_[static_cast<int>(color_sub)].cols[x_sub]   |= bit_sub;

        occupancy_.cols[x_axis] |= bit_axis;
        occupancy_.cols[x_sub]  |= bit_sub;
    }

    __forceinline void dropPiecePairFast(int ax, int sx, int y_axis, int y_sub, Cell color_axis, Cell color_sub) noexcept {
        const uint16_t bit_axis = static_cast<uint16_t>((1U << y_axis) & config::Board::kVisibleColMask);
        const uint16_t bit_sub  = static_cast<uint16_t>((1U << y_sub)  & config::Board::kVisibleColMask);

        boards_[static_cast<int>(color_axis)].cols[ax] |= bit_axis;
        occupancy_.cols[ax] |= bit_axis;

        boards_[static_cast<int>(color_sub)].cols[sx] |= bit_sub;
        occupancy_.cols[sx] |= bit_sub;
    }

    [[nodiscard]] __forceinline const BitBoard& getBitboard(Cell color) const noexcept {
        return boards_[static_cast<int>(color)];
    }

    [[nodiscard]] __forceinline bool isOccupied(int x, int y) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        return (occupancy_.cols[x] >> y) & 1;
    }

    [[nodiscard]] const BitBoard& getOccupied() const noexcept {
        return occupancy_;
    }

  private:
    friend class Gravity;
    friend class Chain;

    static constexpr int toIndex(Cell c) noexcept {
        return static_cast<int>(c);
    }
    std::array<BitBoard, config::Board::kNumColors> boards_{};
    BitBoard occupancy_{};
};

} // namespace puyotan