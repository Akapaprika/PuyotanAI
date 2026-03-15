#pragma once

#include <array>
#include <cstdint>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>

namespace puyotan {

/**
 * BitBoard
 *   128-bit bitfield representing the positions of one piece
 *   color on the 6×14 Puyotan β playing field.
 */
struct BitBoard {
    uint64_t lo = 0;
    uint64_t hi = 0;

    BitBoard() = default;
    constexpr BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}

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

    bool get(int x, int y) const {
        if (x < config::Board::kColsInLo) {
            return (lo >> (x * config::Board::kBitsPerCol + y)) & 1;
        } else {
            return (hi >> ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y)) & 1;
        }
    }

    void set(int x, int y) {
        if (x < config::Board::kColsInLo) {
            lo |=  (1ULL << (x * config::Board::kBitsPerCol + y));
        } else {
            hi |=  (1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
        }
    }

    void clear(int x, int y) {
        if (x < config::Board::kColsInLo) {
            lo &= ~(1ULL << (x * config::Board::kBitsPerCol + y));
        } else {
            hi &= ~(1ULL << ((x - config::Board::kColsInLo) * config::Board::kBitsPerCol + y));
        }
    }

    static constexpr uint64_t kLoMask      = config::Board::kLoMask;
    static constexpr uint64_t kHiMask      = config::Board::kHiMask;
    static constexpr uint64_t kLoSpawnMask = config::Board::kLoSpawnMask;
    static constexpr uint64_t kHiSpawnMask = config::Board::kHiSpawnMask;

    BitBoard shiftUp()    const { return { (lo << 1) & kLoMask, (hi << 1) & kHiMask }; }
    BitBoard shiftDown()  const { return { (lo >> 1) & kLoMask, (hi >> 1) & kHiMask }; }

    BitBoard shiftRight() const {
        uint64_t carry = (lo & (config::Board::kFullLaneMask << (3 * config::Board::kBitsPerCol))) >> 48;
        return {
            (lo << config::Board::kBitsPerCol) & kLoMask,
            ((hi << config::Board::kBitsPerCol) | carry) & kHiMask
        };
    }

    BitBoard shiftLeft() const {
        uint64_t carry = (hi & config::Board::kFullLaneMask) << 48;
        return {
            ((lo >> config::Board::kBitsPerCol) | carry) & kLoMask,
            (hi >> config::Board::kBitsPerCol) & kHiMask
        };
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
