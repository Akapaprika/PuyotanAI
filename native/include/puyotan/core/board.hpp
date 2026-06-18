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
#include <immintrin.h> // BMI2 (_pext_u32) + SSE2/SSE4.1

namespace puyotan {

// 4-bit PDEP LUT for 16-bit column spacing (Bits: 0, 16, 32, 48)
static constexpr uint64_t kPdepLut[16] = {
    0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000010000ULL,
    0x0000000000010001ULL, 0x0000000100000000ULL, 0x0000000100000001ULL,
    0x0000000100010000ULL, 0x0000000100010001ULL, 0x0001000000000000ULL,
    0x0001000000000001ULL, 0x0001000000010000ULL, 0x0001000000010001ULL,
    0x0001000100000000ULL, 0x0001000100000001ULL, 0x0001000100010000ULL,
    0x0001000100010001ULL};

/**
 * @struct BitBoard
 * @brief Represents a single color's puyo positions on a 6x14 field.
 *
 * Uses 128-bit SIMD (__m128i) to store 6 columns of 16-bit lanes.
 * Bits 0-12: Visible rows, Bit 13: Spawn row, Bits 14-15: Unused/Padding.
 */
struct alignas(16) BitBoard {
    union {
        __m128i m128; ///< SIMD 128-bit register
        struct {
            uint64_t lo; ///< Columns 0, 1, 2, 3 (16 bits each)
            uint64_t hi; ///< Columns 4, 5 (16 bits each) + 32-bit padding
        };
    };

    BitBoard() noexcept : m128(_mm_setzero_si128()) {
    }
    constexpr BitBoard(uint64_t l, uint64_t h) noexcept : lo(l), hi(h) {
    }
    BitBoard(__m128i m) noexcept : m128(m) {
    }

    // -----------------------------------------------------------------------
    // Operators -- __forceinline prevents deoptimization on monomorphic hot
    // paths.
    // -----------------------------------------------------------------------
    [[nodiscard]] __forceinline bool
    operator==(const BitBoard& o) const noexcept {
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) != 0;
    }
    [[nodiscard]] __forceinline bool
    operator!=(const BitBoard& o) const noexcept {
        __m128i x = _mm_xor_si128(m128, o.m128);
        return _mm_testz_si128(x, x) == 0;
    }
    [[nodiscard]] __forceinline BitBoard
    operator&(const BitBoard& o) const noexcept {
        return _mm_and_si128(m128, o.m128);
    }
    [[nodiscard]] __forceinline BitBoard
    operator|(const BitBoard& o) const noexcept {
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
    [[nodiscard]] static __forceinline BitBoard
    andNot(const BitBoard& a, const BitBoard& b) noexcept {
        return _mm_andnot_si128(b.m128, a.m128); // result = (~b) & a
    }

    // PTEST (SSE4.1): single instruction -- tests if all bits are zero.
    [[nodiscard]] __forceinline bool empty() const noexcept {
        return _mm_testz_si128(m128, m128) != 0;
    }

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
    static [[nodiscard]] __forceinline BitBoard
    fromColumnMask(uint32_t cols) noexcept {
        const uint64_t mask_lo = kPdepLut[cols & 0x0Fu] * 0xFFFFULL;
        const uint64_t mask_hi = kPdepLut[(cols >> 4) & 0x03u] * 0xFFFFULL;
        return {mask_lo, mask_hi};
    }
    [[nodiscard]] __forceinline int popcount() const noexcept {
        return static_cast<int>(std::popcount(lo) + std::popcount(hi));
    }

    /**
     * Extracts the least significant set bit as a BitBoard (x & -x),
     * implemented entirely in SIMD registers with zero SIMD→GPR round trips.
     *
     * The naive scalar version extracts lo/hi to GPR (vpextrq), computes
     * lo_is_zero_mask with 5 integer ops, then moves back to __m128i —
     * paying a ~5-7 cycle SIMD↔int bypass latency on Intel CPUs.
     *
     * This version stays in XMM land throughout:
     *   1. Compute {lo & -lo, hi & -hi} in parallel with _mm_sub_epi64
     *   2. Detect lo==0 via _mm_cmpeq_epi64
     *   3. Broadcast lo's zero-condition to the hi lane with _mm_shuffle_epi32
     *   4. Build mask {all-1s, lo_zero_cond} with _mm_blend_epi16
     *   5. AND with per-lane LSBs to zero the hi result when lo != 0
     */
    [[nodiscard]] __forceinline BitBoard extractLSB() const noexcept {
        // 1. 各64ビットレーンで個別に LSB を抽出する（lsb = m128 & -m128）
        const __m128i neg = _mm_sub_epi64(_mm_setzero_si128(), m128);
        const __m128i lsb = _mm_and_si128(m128, neg);
    
        // 2. 下位64ビット（lo）がゼロかどうかを判定
        const __m128i cmp = _mm_cmpeq_epi64(m128, _mm_setzero_si128());
    
        // 3. 下位64ビットの比較結果（全0 or 全1）を、上位64ビットへも複製する
        const __m128i lo_zero_mask = _mm_shuffle_epi32(cmp, _MM_SHUFFLE(1, 0, 1, 0));
    
        // 4. マスク処理とブレンド
        //    lo != 0 なら lower レーンの lsb を採用、lo == 0 なら upper レーンの lsb を採用
        const __m128i res_lo = _mm_andnot_si128(lo_zero_mask, lsb);
        const __m128i res_hi = _mm_and_si128(lo_zero_mask, lsb);
    
        // 0xF0 マスクにより、下位64ビット（words 0..3）は res_lo から、
        // 上位64ビット（words 4..7）は res_hi からそれぞれ高速にブレンド
        return _mm_blend_epi16(res_lo, res_hi, 0xF0);
    }

    // -----------------------------------------------------------------------
    // Shift operations -- replaced static local masks with _mm_set_epi64x
    // to bypass MSVC's hidden thread-safety initialization branches and locks.
    // NOTE: The shiftRaw versions do NOT apply the boundary mask, relying on
    // the final '& board' to clean up 'bleeding' bits in row 14/15 or padding.
    // -----------------------------------------------------------------------
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
    [[nodiscard]] __forceinline BitBoard shiftUp() const noexcept {
        return _mm_and_si128(
            shiftUpRaw().m128,
            _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask));
    }
    [[nodiscard]] __forceinline BitBoard shiftDown() const noexcept {
        return _mm_and_si128(
            shiftDownRaw().m128,
            _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask));
    }
    [[nodiscard]] __forceinline BitBoard shiftRight() const noexcept {
        return _mm_and_si128(
            shiftRightRaw().m128,
            _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask));
    }
    [[nodiscard]] __forceinline BitBoard shiftLeft() const noexcept {
        return _mm_and_si128(
            shiftLeftRaw().m128,
            _mm_set_epi64x(config::Board::kHiMask, config::Board::kLoMask));
    }
};

/**
 * @class Board
 * @brief Manages the 6x14 Puyo Puyo playing field using bit-plane
 * representation.
 *
 * This class provides high-performance, branchless operations for querying puyo
 * types, checking column heights, and performing piece placements.
 */
class Board {
  public:
    Board() noexcept = default;
    /**
     * @brief Gets the cell type at the specified grid coordinate.
     * @param x Column index (0-5).
     * @param y Row index (0-13).
     * @return The Cell type at (x, y).
     */
    Cell get(int x, int y) const noexcept;
    /**
     * @brief Sets a specific cell to a color.
     * @param x Column index (0-5).
     * @param y Row index (0-13).
     * @param color The Puyo color to place.
     */
    void set(int x, int y, Cell color) noexcept;
    /**
     * @brief Clears a specific grid cell.
     * @param x Column index (0-5).
     * @param y Row index (0-13).
     */
    void clear(int x, int y) noexcept;
    /**
     * @brief Efficiently sets multiple columns in a single row using a bitmask.
     * @param y Row index (0-13).
     * @param cell The puyo type to set.
     * @param cols_mask 6-bit mask of columns to update.
     */
    void setRowMask(int y, Cell cell, uint32_t cols_mask) noexcept {
        const uint64_t target_lo = kPdepLut[cols_mask & 0x0Fu] << y;
        const uint64_t target_hi = kPdepLut[(cols_mask >> 4) & 0x03u] << y;
        boards_[static_cast<int>(cell)].lo |= target_lo;
        boards_[static_cast<int>(cell)].hi |= target_hi;
        occupancy_.lo |= target_lo;
        occupancy_.hi |= target_hi;
    }
    /**
     * @brief Places a single puyo into the spawn row (Row 13).
     * @param col Target column index (0-5).
     * @param color Color of the puyo.
     */
    void placePiece(int col, Cell color) noexcept;
    /**
     * @brief Calculates the vertical distance a puyo would fall at (x, y).
     * @param x Column index (0-5).
     * @param y Starting row index.
     * @return Number of rows to the nearest obstacle/bottom.
     */
    int getDropDistance(int x, int y) const noexcept;
    /**
     * @brief Returns the height of the puyo stack in a column.
     * @param x Column index (0-5).
     * @return Number of puyos in the column (0-13).
     * @note Performance: O(1) using popcount.
     */
    inline int getColumnHeight(int x) const noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        // BitBoard's lo and hi are contiguous, so we can access them as a
        // 2-element array. x >> 2 (x / 4) maps 0-3 to index 0 (lo) and 4-5 to
        // index 1 (hi).
        const uint64_t val = (&occupancy_.lo)[x >> 2];
        const int shift = (x & 3) << 4; // x % 4 * 16 bits per col
        const uint32_t lane = static_cast<uint32_t>(val >> shift) & 0xFFFFu;
        return static_cast<int>(_mm_popcnt_u32(lane));
    }
    /**
     * @brief Instantly drops a puyo to its final destination, bypassing gravity
     * physics.
     * @param x Target column (0-5).
     * @param y Target row (0-13).
     * @param color Puyo color.
     * @note Primarily used for optimized batch simulation steps.
     */
    inline void dropNewPiece(int x, int y, Cell color) noexcept {
        assert(x >= 0 && x < config::Board::kWidth);
        assert(toIndex(color) >= 0 &&
               toIndex(color) < config::Board::kNumColors);
        const int idx = x >> 2;
        const int col_shift = (x & 3) << 4;
        const int shift = col_shift | y;
        // Branchless visibility mask: y >= 13 is zeroed out by kVisibleColMask
        // (0x1FFF)
        const uint64_t keep_mask =
            static_cast<uint64_t>(config::Board::kVisibleColMask) << col_shift;
        const uint64_t bit = (1ULL << shift) & keep_mask;
        (&boards_[toIndex(color)].lo)[idx] |= bit;
        (&occupancy_.lo)[idx] |= bit;
    }
    /** @brief Retrieves the BitBoard mask for the specified color. */
    [[nodiscard]] inline const BitBoard&
    getBitboard(Cell color) const noexcept {
        return boards_[static_cast<int>(color)];
    }
    /**
     * @brief Fast O(1) occupancy check: true if any puyo occupies (x, y).
     * Preferred over get(x, y) when only empty-or-not is needed (e.g., death
     * check), because it reads a single bit from the occupancy mask rather
     * than iterating over 4 color planes.
     */
    [[nodiscard]] __forceinline bool isOccupied(int x, int y) const noexcept {
        const int idx = x >> 2;
        const int shift = ((x & 3) << 4) | y;
        return ((&occupancy_.lo)[idx] >> shift) & 1;
    }
    /** @brief Manually overwrites the BitBoard for a specific color. */
    void setBitboard(Cell color, const BitBoard& bb) noexcept;
    /** @brief Fully recalculates the occupancy bitmask from all color planes.
     */
    void updateOccupancyFromBoards() noexcept;
    /** @brief Sets the combined occupancy mask. */
    void updateOccupancy(const BitBoard& bb) noexcept {
        occupancy_ = bb;
    }
    /** @brief Returns the combined occupancy mask of all non-empty puyos. */
    const BitBoard& getOccupied() const noexcept {
        return occupancy_;
    }

  private:
    friend class Gravity; // Allow direct lane access for O(1) per-column
                          // gravity
    friend class Chain;
    /**
     * @brief Internal helper to map Cell enum to array index (Red=0...Ojama=4).
     */
    static constexpr int toIndex(Cell c) noexcept {
        return static_cast<int>(c);
    }
    std::array<BitBoard, config::Board::kNumColors>
        boards_{};         ///< Per-color bitmasks
    BitBoard occupancy_{}; ///< Combined occupancy mask
};
} // namespace puyotan
