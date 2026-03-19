#pragma once

#include <array>
#include <cstdint>
#include <bit>
#include <cassert>

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
    [[nodiscard]] __forceinline bool     operator==(const BitBoard& o) const { 
        return _mm_testz_si128(_mm_xor_si128(m128, o.m128), _mm_xor_si128(m128, o.m128)) != 0; 
    }
    [[nodiscard]] __forceinline bool     operator!=(const BitBoard& o) const { 
        return _mm_testz_si128(_mm_xor_si128(m128, o.m128), _mm_xor_si128(m128, o.m128)) == 0; 
    }
    [[nodiscard]] __forceinline BitBoard operator& (const BitBoard& o) const { return _mm_and_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator| (const BitBoard& o) const { return _mm_or_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator~ ()                  const { return _mm_xor_si128(m128, _mm_set1_epi32(-1)); }
    __forceinline BitBoard& operator&=(const BitBoard& o) { m128 = _mm_and_si128(m128, o.m128); return *this; }
    __forceinline BitBoard& operator|=(const BitBoard& o) { m128 = _mm_or_si128(m128, o.m128);  return *this; }

    // PTEST (SSE4.1): single instruction — tests if all bits are zero.
    [[nodiscard]] __forceinline bool empty() const { return _mm_testz_si128(m128, m128) != 0; }

    // Branchless bit access: word = lo/hi, bit offset = (col%4)*16 + row
    [[nodiscard]] __forceinline bool get(int x, int y) const {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        return ((&lo)[x >> 2] >> (((x & 3) << 4) | y)) & 1;
    }
    __forceinline void set(int x, int y) {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        (&lo)[x >> 2] |= (1ULL << (((x & 3) << 4) | y));
    }
    __forceinline void clear(int x, int y) {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        (&lo)[x >> 2] &= ~(1ULL << (((x & 3) << 4) | y));
    }

    [[nodiscard]] __forceinline int popcount() const {
        return static_cast<int>(std::popcount(lo) + std::popcount(hi));
    }

    /**
     * Extracts the least significant set bit as a BitBoard (x & -x).
     * Simplified: if lo==0 and hi==0, hi&-hi = 0 & 0 = 0, so result is {0,0} correctly.
     */
    [[nodiscard]] __forceinline BitBoard extractLSB() const {
        if (lo != 0) {
            return { lo & (0ULL - lo), 0ULL };
        }
        return { 0ULL, hi & (0ULL - hi) };
    }


    // -----------------------------------------------------------------------
    // Shift operations — replaced static local masks with _mm_set_epi64x
    // to bypass MSVC's hidden thread-safety initialization branches and locks.
    // NOTE: The shiftRaw versions do NOT apply the boundary mask, relying on 
    // the final '& board' to clean up 'bleeding' bits in row 14/15 or padding.
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline BitBoard shiftUpRaw()    const { return _mm_slli_epi64(m128, 1);  }
    [[nodiscard]] __forceinline BitBoard shiftDownRaw()  const { return _mm_srli_epi64(m128, 1);  }
    [[nodiscard]] __forceinline BitBoard shiftRightRaw() const { return _mm_slli_si128(m128, 2); }
    [[nodiscard]] __forceinline BitBoard shiftLeftRaw()  const { return _mm_srli_si128(m128, 2); }

    [[nodiscard]] __forceinline BitBoard shiftUp()    const { return _mm_and_si128(shiftUpRaw().m128,    _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftDown()  const { return _mm_and_si128(shiftDownRaw().m128,  _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftRight() const { return _mm_and_si128(shiftRightRaw().m128, _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftLeft()  const { return _mm_and_si128(shiftLeftRaw().m128,  _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
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

    int getDropDistance(int x, int y) const;

    /**
     * O(1) branchless column height query.
     * Uses SIMD popcount under the guarantee that there are no floating puyos.
     */
    inline int getColumnHeight(int x) const {
        assert(x >= 0 && x < config::Board::kWidth);
        uint64_t val = (x < config::Board::kColsInLo) ? occupancy_.lo : occupancy_.hi;
        int shift = (x & 3) << 4; // x % 4 * 16
        return std::popcount(static_cast<uint32_t>(val >> shift) & static_cast<uint32_t>(config::Board::kColMask));
    }

    /**
     * O(1) branchless drop of a single puyo directly to its final destination.
     * Assumes gravity execution will be bypassed.
     */
    inline void dropNewPiece(int x, int y, Cell color) {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        assert(toIndex(color) >= 0 && toIndex(color) < config::Board::kNumColors);
        boards_[toIndex(color)].set(x, y);
        occupancy_.set(x, y);
    }
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
