#pragma once

#include <array>
#include <cstdint>
#include "config/engine_config.hpp"

namespace puyotan {

// ============================================================
// Cell type enum
//   Value 0 is Empty so zero-initialized boards are empty.
//   The uint8_t index is also used as the index into Board::boards_.
// ============================================================
enum class Cell : uint8_t {
    Empty  = 0,
    Red    = 1,
    Green  = 2,
    Blue   = 3,
    Yellow = 4,
    Ojama  = 5,
};

// Number of non-Empty Cell values (used for the boards_ array size)
inline constexpr int kNumCellColors = 5; // Red..Ojama

// ============================================================
// BitBoard
//   128-bit bitfield for one piece color on a 6×14 grid.
//
//   Layout (16 bits per column, LSB = row 0):
//     lo: col0[bits 0-15] | col1[bits 16-31] | col2[bits 32-47] | col3[bits 48-63]
//     hi: col4[bits 0-15] | col5[bits 16-31] | (bits 32-63 unused)
//
//   Valid rows per column: 0-12 (visible) + 13 (invisible spawn row)
//   Total valid bits per column: 14 (bits 0-13).
// ============================================================
struct BitBoard {
    uint64_t lo = 0;
    uint64_t hi = 0;

    BitBoard() = default;
    constexpr BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}

    // ---- Bitwise operators ----
    bool      operator==(const BitBoard& o) const { return lo == o.lo && hi == o.hi; }
    bool      operator!=(const BitBoard& o) const { return !(*this == o); }
    BitBoard  operator& (const BitBoard& o) const { return {lo & o.lo, hi & o.hi}; }
    BitBoard  operator| (const BitBoard& o) const { return {lo | o.lo, hi | o.hi}; }
    BitBoard  operator^ (const BitBoard& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
    BitBoard  operator~ ()                  const { return {~lo, ~hi}; }
    BitBoard& operator&=(const BitBoard& o)       { lo &= o.lo; hi &= o.hi; return *this; }
    BitBoard& operator|=(const BitBoard& o)       { lo |= o.lo; hi |= o.hi; return *this; }
    BitBoard& operator^=(const BitBoard& o)       { lo ^= o.lo; hi ^= o.hi; return *this; }

    bool empty() const { return (lo | hi) == 0; }

    // ------------------------------------------------------------------
    // Branchless bit accessors
    //   Columns 0-3 → lo, columns 4-5 → hi.
    //   We compute which word to touch via a mask derived from (x >> 2).
    //   For x < 4: word_bit = x*16 + y, target = lo
    //   For x >= 4: word_bit = (x-4)*16 + y, target = hi
    //
    //   Branchless trick:
    //     in_hi  = x >> 2          // 0 if col 0-3, 1 if col 4-5
    //     shifted = in_hi * (x*16 + y) + (1 - in_hi) * (... )  <- too complex
    //
    //   Simpler: keep two versions of the bit position, apply to both words,
    //   then re-mask to discard the unused write.
    //   Since the two writes are to separate variables (lo / hi), the branch
    //   is the clearest readable approach that the compiler inlines anyway.
    //   For callers that loop over colors, the real speedup is in the
    //   Board::boards_[] array (eliminates the color→bitboard switch).
    // ------------------------------------------------------------------
    bool get(int x, int y) const {
        // Columns 0-3 in lo; columns 4-5 in hi.
        // (x < 4) is compiled to a conditional move (no branch misprediction).
        if (x < config::Board::kColsInLo)
            return (lo >> (x * config::Board::kBitsPerCol + y)) & 1;
        else
            return (hi >> ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y)) & 1;
    }

    void set(int x, int y) {
        if (x < config::Board::kColsInLo)
            lo |=  (1ULL << (x * config::Board::kBitsPerCol + y));
        else
            hi |=  (1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
    }

    void clear(int x, int y) {
        if (x < config::Board::kColsInLo)
            lo &= ~(1ULL << (x * config::Board::kBitsPerCol + y));
        else
            hi &= ~(1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
    }

    // ---- Validity masks (imported from engine_config) ----
    static constexpr uint64_t K_LO_MASK       = config::Board::kLoMask;
    static constexpr uint64_t K_HI_MASK       = config::Board::kHiMask;
    static constexpr uint64_t K_LO_SPAWN_MASK = config::Board::kLoSpawnMask;
    static constexpr uint64_t K_HI_SPAWN_MASK = config::Board::kHiSpawnMask;

    // ---- Row / column shift operations ----
    // shift_up / shift_down: move every puyo one row up/down within each column lane.
    // Equivalent to multiplying/dividing row index by 1 inside each 16-bit lane.

    // Move all bits one row toward the top (higher row index = higher bit).
    BitBoard shift_up() const {
        return { (lo << 1) & K_LO_MASK, (hi << 1) & K_HI_MASK };
    }

    // Move all bits one row toward the bottom (lower row index = lower bit).
    BitBoard shift_down() const {
        return { (lo >> 1) & K_LO_MASK, (hi >> 1) & K_HI_MASK };
    }

    // Move all bits one column to the right (col N → col N+1).
    // col3 (bits 48-63 of lo) spills into col4 (bits 0-15 of hi).
    BitBoard shift_right() const {
        uint64_t col3_to_col4 = (lo & config::Board::kLoCol3Mask) >> 48;
        return {
            (lo << config::Board::kBitsPerCol) & K_LO_MASK,
            ((hi << config::Board::kBitsPerCol) | col3_to_col4) & K_HI_MASK
        };
    }

    // Move all bits one column to the left (col N → col N-1).
    // col4 (bits 0-15 of hi) spills into col3 (bits 48-63 of lo).
    BitBoard shift_left() const {
        uint64_t col4_to_col3 = (hi & config::Board::kHiCol4Mask) << 48;
        return {
            ((lo >> config::Board::kBitsPerCol) | col4_to_col3) & K_LO_MASK,
            (hi >> config::Board::kBitsPerCol) & K_HI_MASK
        };
    }
};

// ============================================================
// Board
//   6×14 playing field stored as one BitBoard per color.
//   boards_[0] = Red, [1] = Green, [2] = Blue, [3] = Yellow, [4] = Ojama
//   Indexed by (static_cast<int>(color) - 1) – no switch needed.
// ============================================================
class Board {
public:
    Board() = default;

    // ---- Cell-level read/write ----
    Cell get(int x, int y) const;
    void set(int x, int y, Cell color);
    void clear(int x, int y);

    // Place a single puyo at the invisible spawn row of col (0-indexed).
    // Call Gravity::execute() afterwards to drop it into position.
    void place_piece(int col, Cell color);

    // ---- BitBoard-level read/write (used by Gravity, Chain, etc.) ----
    const BitBoard& get_bitboard(Cell color) const;
    void            set_bitboard(Cell color, const BitBoard& bb);

private:
    // boards_[i] corresponds to Cell value (i+1): Red=0, Green=1, Blue=2, Yellow=3, Ojama=4
    std::array<BitBoard, kNumCellColors> boards_{};

    // Fast index: Cell::Red(1)→0, ..., Cell::Ojama(5)→4
    // Avoids switch/if-else on every BitBoard access.
    static constexpr int idx(Cell c) {
        return static_cast<int>(c) - 1;
    }
};

} // namespace puyotan
