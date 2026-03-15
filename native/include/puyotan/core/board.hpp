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
#include <immintrin.h> // BMI2 (_pext_u32) + SSE2/SSE4.1

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

    // -----------------------------------------------------------------------
    // Operators — __forceinline prevents deoptimization on monomorphic hot paths.
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline bool     operator==(const BitBoard& o) const { return lo == o.lo && hi == o.hi; }
    [[nodiscard]] __forceinline bool     operator!=(const BitBoard& o) const { return lo != o.lo || hi != o.hi; }
    [[nodiscard]] __forceinline BitBoard operator& (const BitBoard& o) const { return _mm_and_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator| (const BitBoard& o) const { return _mm_or_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator^ (const BitBoard& o) const { return _mm_xor_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator~ ()                  const { return _mm_xor_si128(m128, _mm_set1_epi32(-1)); }
    __forceinline BitBoard& operator&=(const BitBoard& o) { m128 = _mm_and_si128(m128, o.m128); return *this; }
    __forceinline BitBoard& operator|=(const BitBoard& o) { m128 = _mm_or_si128(m128, o.m128);  return *this; }
    __forceinline BitBoard& operator^=(const BitBoard& o) { m128 = _mm_xor_si128(m128, o.m128); return *this; }

    // PTEST (SSE4.1): single instruction — tests if all bits are zero.
    [[nodiscard]] __forceinline bool empty() const { return _mm_testz_si128(m128, m128) != 0; }

    // Branchless bit access: word = lo/hi, bit offset = (col%4)*16 + row
    [[nodiscard]] __forceinline bool get(int x, int y) const {
        return ((&lo)[x >> 2] >> (((x & 3) << 4) | y)) & 1;
    }
    __forceinline void set(int x, int y) {
        (&lo)[x >> 2] |= (1ULL << (((x & 3) << 4) | y));
    }
    __forceinline void clear(int x, int y) {
        (&lo)[x >> 2] &= ~(1ULL << (((x & 3) << 4) | y));
    }

    [[nodiscard]] __forceinline int popcount() const {
#ifdef _MSC_VER
        return static_cast<int>(__popcnt64(lo) + __popcnt64(hi));
#else
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
#endif
    }

    /**
     * Extracts the least significant set bit as a BitBoard (x & -x).
     * Simplified: if lo==0 and hi==0, hi&-hi = 0 & 0 = 0, so result is {0,0} correctly.
     */
    [[nodiscard]] __forceinline BitBoard extractLSB() const {
        if (lo != 0) {
            return { lo & static_cast<uint64_t>(-static_cast<int64_t>(lo)), 0ULL };
        }
        return { 0ULL, hi & static_cast<uint64_t>(-static_cast<int64_t>(hi)) };
    }

    // -----------------------------------------------------------------------
    // Static masks — compile-time constants
    // -----------------------------------------------------------------------
    static BitBoard kLoMask()   { return { config::Board::kLoMask, 0 }; }
    static BitBoard kHiMask()   { return { 0, config::Board::kHiMask }; }
    static BitBoard kFullMask() { return { config::Board::kLoMask, config::Board::kHiMask }; }

    // -----------------------------------------------------------------------
    // Shift operations — kFullMask stored as static local for guaranteed
    // register-caching by the compiler (no reconstruction per call).
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline BitBoard shiftUp() const {
        static const __m128i kMask = _mm_set_epi64x(
            static_cast<int64_t>(config::Board::kHiMask),
            static_cast<int64_t>(config::Board::kLoMask));
        return _mm_and_si128(_mm_slli_epi64(m128, 1), kMask);
    }

    [[nodiscard]] __forceinline BitBoard shiftDown() const {
        static const __m128i kMask = _mm_set_epi64x(
            static_cast<int64_t>(config::Board::kHiMask),
            static_cast<int64_t>(config::Board::kLoMask));
        return _mm_and_si128(_mm_srli_epi64(m128, 1), kMask);
    }

    [[nodiscard]] __forceinline BitBoard shiftRight() const {
        // Shift whole register left by 2 bytes (= one 16-bit column lane right in display).
        static const __m128i kMask = _mm_set_epi64x(
            static_cast<int64_t>(config::Board::kHiMask),
            static_cast<int64_t>(config::Board::kLoMask));
        return _mm_and_si128(_mm_slli_si128(m128, 2), kMask);
    }

    [[nodiscard]] __forceinline BitBoard shiftLeft() const {
        // Shift whole register right by 2 bytes.
        static const __m128i kMask = _mm_set_epi64x(
            static_cast<int64_t>(config::Board::kHiMask),
            static_cast<int64_t>(config::Board::kLoMask));
        return _mm_and_si128(_mm_srli_si128(m128, 2), kMask);
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
    friend class Gravity; // Allow direct lane access for O(1) per-column gravity

    std::array<BitBoard, config::Board::kNumColors> boards_{};
    BitBoard occupancy_{};

    static constexpr int toIndex(Cell c) { return static_cast<int>(c); }
};

} // namespace puyotan
