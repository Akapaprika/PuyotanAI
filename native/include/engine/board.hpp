#pragma once

#include <array>
#include <cstdint>
#include "config/engine_config.hpp"

namespace puyotan {

enum class Cell : uint8_t {
    Empty = 0,
    Red,
    Green,
    Blue,
    Yellow,
    Ojama
};

struct BitBoard {
    uint64_t lo = 0; // col 1-4 (16bit per col)
    uint64_t hi = 0; // col 5-6 (16bit per col, 32bit unused)

    BitBoard() = default;
    BitBoard(uint64_t l, uint64_t h) : lo(l), hi(h) {}

    bool operator==(const BitBoard& other) const { return lo == other.lo && hi == other.hi; }
    bool operator!=(const BitBoard& other) const { return !(*this == other); }
    BitBoard operator&(const BitBoard& other) const { return {lo & other.lo, hi & other.hi}; }
    BitBoard operator|(const BitBoard& other) const { return {lo | other.lo, hi | other.hi}; }
    BitBoard operator^(const BitBoard& other) const { return {lo ^ other.lo, hi ^ other.hi}; }
    BitBoard operator~() const { return {~lo, ~hi}; }

    BitBoard& operator&=(const BitBoard& other) { lo &= other.lo; hi &= other.hi; return *this; }
    BitBoard& operator|=(const BitBoard& other) { lo |= other.lo; hi |= other.hi; return *this; }
    BitBoard& operator^=(const BitBoard& other) { lo ^= other.lo; hi ^= other.hi; return *this; }

    bool empty() const { return lo == 0 && hi == 0; }
    
    // Check if bit is set at (x, y) where x is 0-5, y is 0-15
    bool get(int x, int y) const {
        if (x < 4) {
            return (lo & (1ULL << (x * 16 + y))) != 0;
        } else {
            return (hi & (1ULL << ((x - 4) * 16 + y))) != 0;
        }
    }

    // Set bit at (x, y)
    void set(int x, int y) {
        if (x < 4) {
            lo |= (1ULL << (x * 16 + y));
        } else {
            hi |= (1ULL << ((x - 4) * 16 + y));
        }
    }

    // Clear bit at (x, y)
    void clear(int x, int y) {
        if (x < 4) {
            lo &= ~(1ULL << (x * 16 + y));
        } else {
            hi &= ~(1ULL << ((x - 4) * 16 + y));
        }
    }

    // Constants for 6x13 field (13th row used for hidden row)
    static constexpr uint64_t K_LO_MASK = 0x1FFF1FFF1FFF1FFF; // 13 rows per col (0-12)
    static constexpr uint64_t K_HI_MASK = 0x000000001FFF1FFF;
};

inline int index(int x, int y) {
    return (x << 4) + y;
}

class Board {
public:
    Board();

    Cell get(int x, int y) const;
    void set(int x, int y, Cell color);
    
    // Remove puyo at (x, y)
    void clear(int x, int y);

    // Get BitBoard for a specific color
    const BitBoard& get_bitboard(Cell color) const;

private:
    BitBoard board_red_;
    BitBoard board_green_;
    BitBoard board_blue_;
    BitBoard board_yellow_;
    BitBoard board_ojama_;
};

}
