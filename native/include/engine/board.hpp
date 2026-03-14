#pragma once

#include <array>
#include <cstdint>
#include "config/engine_config.hpp"

namespace puyotan {

// ============================================================
// Cell
//   Represents the occupant of a single board cell.
//
//   Values Red(0)..Ojama(4) are valid array indices into
//   Board::boards_[] — no offset arithmetic required.
//
//   Empty is a sentinel (0xFF) that is never stored in any
//   BitBoard plane; it is only returned by Board::get() when
//   no color bit is set at a given position.
//   Using 0xFF (instead of 0) means zero-initialised local
//   Cell variables are clearly invalid, which aids debugging.
// ============================================================
enum class Cell : uint8_t {
    Red    = 0,  // boards_[0]
    Green  = 1,  // boards_[1]
    Blue   = 2,  // boards_[2]
    Yellow = 3,  // boards_[3]
    Ojama  = 4,  // boards_[4]
    Empty  = 0xFF,  // sentinel — never stored in BitBoard planes
};

// ============================================================
// BitBoard
//   128-bit bitfield representing the positions of one piece
//   color on the 6×14 Puyotan β playing field.
//
//   Memory layout (16 bits per column, LSB = row 0):
//     lo (uint64_t): col0[0-15] | col1[16-31] | col2[32-47] | col3[48-63]
//     hi (uint64_t): col4[0-15] | col5[16-31] | (bits 32-63 unused)
//
//   Valid row range per column: 0-12 (visible) + 13 (invisible spawn row).
//   Only 14 of the 16 bits in each column lane are ever set.
// ============================================================
struct BitBoard {
    uint64_t lo = 0;
    uint64_t hi = 0;

    BitBoard() = default;
    constexpr BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}

    // ---- Bitwise operators — operate on both 64-bit halves simultaneously ----
    bool      operator==(const BitBoard& o) const { return lo == o.lo && hi == o.hi; }
    bool      operator!=(const BitBoard& o) const { return !(*this == o); }
    BitBoard  operator& (const BitBoard& o) const { return {lo & o.lo, hi & o.hi}; }
    BitBoard  operator| (const BitBoard& o) const { return {lo | o.lo, hi | o.hi}; }
    BitBoard  operator^ (const BitBoard& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
    BitBoard  operator~ ()                  const { return {~lo, ~hi}; }
    BitBoard& operator&=(const BitBoard& o)       { lo &= o.lo; hi &= o.hi; return *this; }
    BitBoard& operator|=(const BitBoard& o)       { lo |= o.lo; hi |= o.hi; return *this; }
    BitBoard& operator^=(const BitBoard& o)       { lo ^= o.lo; hi ^= o.hi; return *this; }

    // Returns true if no bits are set (board plane is empty).
    bool empty() const { return (lo | hi) == 0; }

    // ---- Single-cell accessors ----
    // x: column index [0, 5],  y: row index [0, 13].
    // Columns 0-3 are packed in lo; columns 4-5 are packed in hi.
    // The branch is a compare-and-select; modern compilers emit a CMOV
    // (no branch misprediction penalty in a tight inner loop).

    /// Returns true if the bit at (x, y) is set.
    bool get(int x, int y) const {
        if (x < config::Board::kColsInLo) {
            return (lo >> (x * config::Board::kBitsPerCol + y)) & 1;
        } else {
            return (hi >> ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y)) & 1;
        }
    }

    /// Sets the bit at (x, y).
    void set(int x, int y) {
        if (x < config::Board::kColsInLo) {
            lo |=  (1ULL << (x * config::Board::kBitsPerCol + y));
        } else {
            hi |=  (1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
        }
    }

    /// Clears the bit at (x, y).
    void clear(int x, int y) {
        if (x < config::Board::kColsInLo) {
            lo &= ~(1ULL << (x * config::Board::kBitsPerCol + y));
        } else {
            hi &= ~(1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
        }
    }

    // ---- Validity masks (aliases from engine_config, named kCamelCase) ----
    static constexpr uint64_t kLoMask      = config::Board::kLoMask;
    static constexpr uint64_t kHiMask      = config::Board::kHiMask;
    static constexpr uint64_t kLoSpawnMask = config::Board::kLoSpawnMask;
    static constexpr uint64_t kHiSpawnMask = config::Board::kHiSpawnMask;

    // ---- Row-shift operations ----
    // Used by Gravity and Chain detection to find adjacent cells.

    /// Moves every bit one row upward (row index +1 = higher bit in each lane).
    BitBoard shiftUp()    const { return { (lo << 1) & kLoMask, (hi << 1) & kHiMask }; }

    /// Moves every bit one row downward (row index -1 = lower bit in each lane).
    BitBoard shiftDown()  const { return { (lo >> 1) & kLoMask, (hi >> 1) & kHiMask }; }

    // ---- Column-shift operations ----
    // col3 (lo bits 48-63) and col4 (hi bits 0-15) are adjacent and
    // must carry bits across the lo/hi boundary.

    /// Moves every bit one column to the right (col N → col N+1).
    BitBoard shiftRight() const {
        // col3 overflows from lo into the bottom lane of hi
        uint64_t carry = (lo & config::Board::kLoCol3Mask) >> 48;
        return {
            (lo << config::Board::kBitsPerCol) & kLoMask,
            ((hi << config::Board::kBitsPerCol) | carry) & kHiMask
        };
    }

    /// Moves every bit one column to the left (col N → col N-1).
    BitBoard shiftLeft() const {
        // col4 underflows from hi into the top lane of lo
        uint64_t carry = (hi & config::Board::kHiCol4Mask) << 48;
        return {
            ((lo >> config::Board::kBitsPerCol) | carry) & kLoMask,
            (hi >> config::Board::kBitsPerCol) & kHiMask
        };
    }
};

// ============================================================
// Board
//   Puyotan β 6×14 playing field.
//   Internally stores one BitBoard per color in boards_[].
//   boards_[static_cast<int>(color)] — direct cast, no offset.
//
//   Coordinate system (0-indexed):
//     x: column 0 (left) … 5 (right)
//     y: row    0 (bottom) … 12 (visible top) … 13 (spawn row, invisible)
// ============================================================
class Board {
public:
    Board() = default;

    // ---- Cell-level interface ----

    /// Returns the color at (x, y), or Cell::Empty if unoccupied.
    /// Scans all 5 color planes; typically unrolled by the compiler.
    Cell get(int x, int y) const;

    /// Places color at (x, y).
    /// Clears any existing occupant first (safe to overwrite).
    /// Setting Cell::Empty is a no-op after clearing.
    void set(int x, int y, Cell color);

    /// Clears all color planes at (x, y) unconditionally.
    /// O(kNumColors) but trivially branch-free.
    void clear(int x, int y);

    /// Places a single puyo at the invisible spawn row (y = kSpawnRow)
    /// of the given column (0-indexed).
    /// Call Gravity::execute() afterwards to drop it into position.
    /// Out-of-range columns are silently ignored.
    void placePiece(int col, Cell color);

    // ---- BitBoard-level interface (for Gravity, Chain, Evaluator) ----

    /// Returns the BitBoard for color.  Cell::Empty is undefined behaviour.
    const BitBoard& getBitboard(Cell color) const;

    /// Overwrites the BitBoard for color.  Cell::Empty is undefined behaviour.
    /// Also updates the global occupancy mask.
    void setBitboard(Cell color, const BitBoard& bb);

    /// Returns a combined BitBoard representing all occupied cells.
    /// Includes Ojama and all 4 colors.
    const BitBoard& getOccupied() const {
        return occupancy_;
    }

private:
    // Direct index: boards_[static_cast<int>(Cell::Red)]   = Red plane
    //               boards_[static_cast<int>(Cell::Ojama)] = Ojama plane
    // Cell values Red(0)..Ojama(4) are contiguous — no offset needed.
    std::array<BitBoard, config::Board::kNumColors> boards_{};
    
    // Combined occupancy mask for performance (Gravity, get).
    BitBoard occupancy_{};

    // Converts a Cell to the boards_[] index.
    // Cell values are already 0-based so this is just a cast.
    static constexpr int toIndex(Cell c) {
        return static_cast<int>(c);
    }
};

} // namespace puyotan
