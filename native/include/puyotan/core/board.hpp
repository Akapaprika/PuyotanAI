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

    BitBoard() noexcept : m128(_mm_setzero_si128()) {}
    constexpr BitBoard(uint64_t l, uint64_t h) noexcept : lo(l), hi(h) {}
    BitBoard(__m128i m) noexcept : m128(m) {}

    // -----------------------------------------------------------------------
    // Operators — __forceinline prevents deoptimization on monomorphic hot paths.
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline bool     operator==(const BitBoard& o) const noexcept { 
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) != 0; 
    }
    [[nodiscard]] __forceinline bool     operator!=(const BitBoard& o) const noexcept { 
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) == 0; 
    }
    [[nodiscard]] __forceinline BitBoard operator& (const BitBoard& o) const noexcept { return _mm_and_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator| (const BitBoard& o) const noexcept { return _mm_or_si128(m128, o.m128); }
    [[nodiscard]] __forceinline BitBoard operator~ ()                  const noexcept { return _mm_xor_si128(m128, _mm_set1_epi32(-1)); }
    __forceinline BitBoard& operator&=(const BitBoard& o) noexcept { m128 = _mm_and_si128(m128, o.m128); return *this; }
    __forceinline BitBoard& operator|=(const BitBoard& o) noexcept { m128 = _mm_or_si128(m128, o.m128);  return *this; }
    __forceinline BitBoard& andNot(const BitBoard& o) noexcept     { m128 = _mm_andnot_si128(o.m128, m128); return *this; }
    [[nodiscard]] static __forceinline BitBoard andNot(const BitBoard& a, const BitBoard& b) noexcept { 
        return _mm_andnot_si128(b.m128, a.m128); // result = (~b) & a
    }

    // PTEST (SSE4.1): single instruction — tests if all bits are zero.
    [[nodiscard]] __forceinline bool empty() const noexcept { return _mm_testz_si128(m128, m128) != 0; }

    // Branchless bit access: word = lo/hi, bit offset = (col%4)*16 + row
    [[nodiscard]] __forceinline bool get(int x, int y) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        int idx = x >> 2;
        int shift = ((x & 3) << 4) | y;
        return ((&lo)[idx] >> shift) & 1;
    }
    __forceinline void set(int x, int y) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        int idx = x >> 2;
        int shift = ((x & 3) << 4) | y;
        (&lo)[idx] |= (1ULL << shift);
    }
    __forceinline void clear(int x, int y) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        int idx = x >> 2;
        int shift = ((x & 3) << 4) | y;
        (&lo)[idx] &= ~(1ULL << shift);
    }

    static [[nodiscard]] __forceinline BitBoard fromColumnMask(uint32_t cols) noexcept {
        uint64_t mask_lo = _pdep_u64(cols & 0x0F, 0x0001000100010001ULL) * 0xFFFFULL;
        uint64_t mask_hi = _pdep_u64((cols >> 4) & 0x03, 0x0000000000010001ULL) * 0xFFFFULL;
        return {mask_lo, mask_hi};
    }

    [[nodiscard]] __forceinline int popcount() const noexcept {
        return static_cast<int>(std::popcount(lo) + std::popcount(hi));
    }

    /**
     * Extracts the least significant set bit as a BitBoard (x & -x).
     * Simplified: if lo==0 and hi==0, hi&-hi = 0 & 0 = 0, so result is {0,0} correctly.
     */
    [[nodiscard]] __forceinline BitBoard extractLSB() const noexcept {
        uint64_t new_lo = lo & (0ULL - lo);
        // If we simply apply extractLSB to both lo and hi, we might extract two bits, which is a bug.
        // Therefore, we must only process hi if lo is empty.
        // Using a ternary operator here allows the compiler to use a Conditional Move (CMOV)
        // instruction instead of a branch (JMP), effectively eliminating branch misprediction penalties.
        uint64_t new_hi = (lo == 0) ? (hi & (0ULL - hi)) : 0ULL;
        return { new_lo, new_hi };
    }


    // -----------------------------------------------------------------------
    // Shift operations — replaced static local masks with _mm_set_epi64x
    // to bypass MSVC's hidden thread-safety initialization branches and locks.
    // NOTE: The shiftRaw versions do NOT apply the boundary mask, relying on 
    // the final '& board' to clean up 'bleeding' bits in row 14/15 or padding.
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline BitBoard shiftUpRaw()    const noexcept { return _mm_slli_epi64(m128, 1);  }
    [[nodiscard]] __forceinline BitBoard shiftDownRaw()  const noexcept { return _mm_srli_epi64(m128, 1);  }
    [[nodiscard]] __forceinline BitBoard shiftRightRaw() const noexcept { return _mm_slli_si128(m128, 2); }
    [[nodiscard]] __forceinline BitBoard shiftLeftRaw()  const noexcept { return _mm_srli_si128(m128, 2); }

    [[nodiscard]] __forceinline BitBoard shiftUp()    const noexcept { return _mm_and_si128(shiftUpRaw().m128,    _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftDown()  const noexcept { return _mm_and_si128(shiftDownRaw().m128,  _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftRight() const noexcept { return _mm_and_si128(shiftRightRaw().m128, _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
    [[nodiscard]] __forceinline BitBoard shiftLeft()  const noexcept { return _mm_and_si128(shiftLeftRaw().m128,  _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask)); }
};

/**
 * Board
 *   Puyotan β 6×14 playing field using BitBoard planes.
 */
class Board {
public:
    Board() noexcept = default;

    Cell get(int x, int y) const noexcept;
    void set(int x, int y, Cell color) noexcept;
    void clear(int x, int y) noexcept;

    void setRowMask(int y, Cell cell, uint32_t cols_mask) noexcept {
        const uint64_t target_lo = _pdep_u64(cols_mask & 0x0Fu, 0x0001000100010001ULL) << y;
        const uint64_t target_hi = _pdep_u64((cols_mask >> 4) & 0x03u, 0x0000000000010001ULL) << y;

        boards_[static_cast<int>(cell)].lo |= target_lo;
        boards_[static_cast<int>(cell)].hi |= target_hi;
        occupancy_.lo |= target_lo;
        occupancy_.hi |= target_hi;
    }

    void placePiece(int col, Cell color) noexcept;

    int getDropDistance(int x, int y) const noexcept;

    /**
     * O(1) branchless column height query.
     * Uses SIMD popcount under the guarantee that there are no floating puyos.
     */
    inline int getColumnHeight(int x) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        uint64_t val = (x < config::Board::kColsInLo) ? occupancy_.lo : occupancy_.hi;
        int shift = (x & 3) << 4; // x % 4 * 16
        const uint32_t lane = (static_cast<uint32_t>(val >> shift)) & ((1u << config::Board::kBitsPerCol) - 1);
        return static_cast<int>(_mm_popcnt_u32(lane));
    }
    /**
     * O(1) branchless drop of a single puyo directly to its final destination.
     * Assumes gravity execution will be bypassed.
     */
    inline void dropNewPiece(int x, int y, Cell color) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(y >= 0 && y < config::Board::kHeight + 1);
        assert(toIndex(color) >= 0 && toIndex(color) < config::Board::kNumColors);
        boards_[toIndex(color)].set(x, y);
        occupancy_.set(x, y);
    }
    [[nodiscard]] const BitBoard& getBitboard(Cell color) const noexcept;
    void setBitboard(Cell color, const BitBoard& bb) noexcept;

    // Full recalculation of occupancy_ from all color boards (O(N))
    void updateOccupancyFromBoards() noexcept;

    void updateOccupancy(const BitBoard& bb) noexcept { occupancy_ = bb; }
    const BitBoard& getOccupied() const noexcept { return occupancy_; }

private:
    friend class Gravity; // Allow direct lane access for O(1) per-column gravity

    std::array<BitBoard, config::Board::kNumColors> boards_{};
    BitBoard occupancy_{};

    static constexpr int toIndex(Cell c) { return static_cast<int>(c); }
};

} // namespace puyotan
