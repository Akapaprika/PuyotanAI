#pragma once

#include <array>
#include <cstdint>

#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace puyotan {

/**
 * BitBoard
 *   128-bit SIMD-optimized bitfield representing the positions of one piece
 *   color on the 6×14 Puyotan β playing field.
 */
struct alignas(16) BitBoard {
    union {
        __m128i m128;
        struct { uint64_t lo, hi; };
    };

    BitBoard() : m128(_mm_setzero_si128()) {}
    constexpr BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    BitBoard(__m128i m) : m128(m) {}

    bool      operator==(const BitBoard& o) const { 
        // _mm_testc_si128 can be used for equality too, but lo/hi comparison is fine.
        return lo == o.lo && hi == o.hi; 
    }
    bool      operator!=(const BitBoard& o) const { return !(*this == o); }
    BitBoard  operator& (const BitBoard& o) const { return _mm_and_si128(m128, o.m128); }
    BitBoard  operator| (const BitBoard& o) const { return _mm_or_si128(m128, o.m128); }
    BitBoard  operator^ (const BitBoard& o) const { return _mm_xor_si128(m128, o.m128); }
    BitBoard  operator~ ()                  const { 
        return _mm_xor_si128(m128, _mm_set1_epi32(0xFFFFFFFF)); 
    }
    BitBoard& operator&=(const BitBoard& o)       { m128 = _mm_and_si128(m128, o.m128); return *this; }
    BitBoard& operator|=(const BitBoard& o)       { m128 = _mm_or_si128(m128, o.m128); return *this; }
    BitBoard& operator^=(const BitBoard& o)       { m128 = _mm_xor_si128(m128, o.m128); return *this; }

    bool empty() const { 
        // PTEST (SSE4.1) is the fastest way to check for all-zeros.
        return _mm_testz_si128(m128, m128);
    }

    bool get(int x, int y) const {
        const uint64_t* data = &lo;
        // word_idx = x / 4 (x >> 2), local_col = x % 4 (x & 3)
        // bit_offset = local_col * 16 + y
        return (data[x >> 2] >> (((x & 3) << 4) | y)) & 1;
    }

    void set(int x, int y) {
        uint64_t* data = &lo;
        data[x >> 2] |= (1ULL << (((x & 3) << 4) | y));
    }

    void clear(int x, int y) {
        uint64_t* data = &lo;
        data[x >> 2] &= ~(1ULL << (((x & 3) << 4) | y));
    }

    int popcount() const {
#ifdef _MSC_VER
        return (int)(__popcnt64(lo) + __popcnt64(hi));
#else
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
#endif
    }

    /**
     * Extracts the least significant bit as a BitBoard.
     */
    BitBoard extractLSB() const {
        if (lo != 0) {
            return { lo & (uint64_t)(-(int64_t)lo), 0 };
        } else if (hi != 0) {
            return { 0, hi & (uint64_t)(-(int64_t)hi) };
        }
        return { 0, 0 };
    }

    static BitBoard kLoMask()      { return { config::Board::kLoMask, 0 }; }
    static BitBoard kHiMask()      { return { 0, config::Board::kHiMask }; }
    static BitBoard kFullMask()    { return { config::Board::kLoMask, config::Board::kHiMask }; }

    BitBoard shiftUp() const { 
        // Parallel 64-bit shifts (no scalar fallback).
        return _mm_and_si128(
            _mm_slli_epi64(m128, 1),
            kFullMask().m128
        );
    }

    BitBoard shiftDown() const {
        // Parallel 64-bit shifts.
        return _mm_and_si128(
            _mm_srli_epi64(m128, 1),
            kFullMask().m128
        );
    }

    BitBoard shiftRight() const {
        // Shift entire 128-bit register by 16 bits (2 bytes).
        return _mm_and_si128(
            _mm_slli_si128(m128, 2),
            kFullMask().m128
        );
    }

    BitBoard shiftLeft() const {
        // Shift entire 128-bit register by 16 bits (2 bytes) right.
        return _mm_and_si128(
            _mm_srli_si128(m128, 2),
            kFullMask().m128
        );
    }
};

/**
 * Board
 *   Puyotan β 6×14 playing field using BitBoard planes.
 */
class Board {
public:
    Board() = default;

    Cell get(int x, int y) const;
    void set(int x, int y, Cell color);
    void clear(int x, int y);

    void placePiece(int col, Cell color);

    /**
     * Calculates the number of rows a puyo at (x, y) would fall.
     * Matches Puyotan β's independent distance calculation.
     */
    int getDropDistance(int x, int y) const;
    const BitBoard& getBitboard(Cell color) const;
    void setBitboard(Cell color, const BitBoard& bb, bool update_occupancy = true);

    void updateOccupancy(const BitBoard& bb) { occupancy_ = bb; }
    const BitBoard& getOccupied() const { return occupancy_; }

private:
    std::array<BitBoard, config::Board::kNumColors> boards_{};
    BitBoard occupancy_{};

    static constexpr int toIndex(Cell c) { return static_cast<int>(c); }
};

} // namespace puyotan
