#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

namespace puyotan {

static constexpr uint64_t kPdepLut[16] = {
    0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000010000ULL,
    0x0000000000010001ULL, 0x0000000100000000ULL, 0x0000000100000001ULL,
    0x0000000100010000ULL, 0x0000000100010001ULL, 0x0001000000000000ULL,
    0x0001000000000001ULL, 0x0001000000010000ULL, 0x0001000000010001ULL,
    0x0001000100000000ULL, 0x0001000100000001ULL, 0x0001000100010000ULL,
    0x0001000100010001ULL};

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

    [[nodiscard]] static __forceinline BitBoard fromColumnMask(uint32_t cols) noexcept {
        const uint64_t mask_lo = kPdepLut[cols & 0x0Fu] * 0xFFFFULL;
        const uint64_t mask_hi = kPdepLut[(cols >> 4) & 0x03u] * 0xFFFFULL;
        return {mask_lo, mask_hi};
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
    Board() noexcept = default;

    [[nodiscard]] __forceinline Cell get(int x, int y) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        const int b1  = (boards_[1].cols[x] >> y) & 1;
        const int b2  = (boards_[2].cols[x] >> y) & 1;
        const int b3  = (boards_[3].cols[x] >> y) & 1;
        const int b4  = (boards_[4].cols[x] >> y) & 1;
        const int occ = (occupancy_.cols[x] >> y) & 1;
        return static_cast<Cell>(b1 + (b2 * 2) + (b3 * 3) + (b4 * 4) + (5 * (1 - occ)));
    }

    __forceinline void set(int x, int y, Cell color) noexcept {
        assert(color != Cell::Empty);
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        const uint16_t bit = static_cast<uint16_t>(1U << y);
        const uint16_t clear_mask = static_cast<uint16_t>(~bit);
        for (auto& bb : boards_) {
            bb.cols[x] &= clear_mask;
        }
        boards_[toIndex(color)].cols[x] |= bit;
        occupancy_.cols[x] |= bit;
    }

    __forceinline void clear(int x, int y) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        const uint16_t clear_mask = static_cast<uint16_t>(~(1U << y));
        for (auto& bb : boards_) {
            bb.cols[x] &= clear_mask;
        }
        occupancy_.cols[x] &= clear_mask;
    }

    void setRowMask(int y, Cell cell, uint32_t cols_mask) noexcept {
        const uint64_t target_lo = kPdepLut[cols_mask & 0x0Fu] << y;
        const uint64_t target_hi = kPdepLut[(cols_mask >> 4) & 0x03u] << y;
        boards_[static_cast<int>(cell)].lo |= target_lo;
        boards_[static_cast<int>(cell)].hi |= target_hi;
        occupancy_.lo |= target_lo;
        occupancy_.hi |= target_hi;
    }

    /// ★ 完全インライン化: 発生行(13)への配置を 1命令の 16bit OR にコンパイル
    __forceinline void placePiece(int col, Cell color) noexcept {
        assert(col >= 0 && col < config::Board::kWidth);
        set(col, config::Board::kSpawnRow, color);
    }

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

        const uint32_t bit1_axis = _blsi_u32(~lane_axis);
        const uint32_t bit1_sub  = _blsi_u32(~lane_sub);

        out_h_axis = std::countr_zero(bit1_axis);
        out_h_sub  = std::countr_zero(bit1_sub);

        const bool is_same_col = (x_axis == x_sub);
        const int use_2nd_axis = static_cast<int>(is_same_col && (r == Rotation::Down));
        const int use_2nd_sub  = static_cast<int>(is_same_col && (r == Rotation::Up));

        const uint16_t bit_axis = static_cast<uint16_t>((bit1_axis << use_2nd_axis) & config::Board::kVisibleColMask);
        const uint16_t bit_sub  = static_cast<uint16_t>((bit1_sub << use_2nd_sub) & config::Board::kVisibleColMask);

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

    void setBitboard(Cell color, const BitBoard& bb) noexcept {
        boards_[toIndex(color)] = bb;
    }
    void updateOccupancyFromBoards() noexcept {
        const __m128i or01   = _mm_or_si128(boards_[0].m128, boards_[1].m128);
        const __m128i or23   = _mm_or_si128(boards_[2].m128, boards_[3].m128);
        const __m128i or0123 = _mm_or_si128(or01, or23);
        occupancy_.m128      = _mm_or_si128(or0123, boards_[4].m128);
    }
    void updateOccupancy(const BitBoard& bb) noexcept {
        occupancy_ = bb;
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