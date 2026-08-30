#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>

namespace puyotan::search {

class Zobrist {
public:
    // 5色 (4色 + おじゃま) × 6列 × 16段 = 480 entries (3,840 bytes, L1D Cache常駐)
    inline static uint64_t table[config::Board::kNumColors][config::Board::kWidth][16];
    inline static bool initialized = false;

    static void init() noexcept {
        if (initialized) [[likely]] return;
        // 決定論的PRNG (SplitMix64)
        uint64_t x = 0x9E3779B97F4A7C15ULL;
        for (int c = 0; c < config::Board::kNumColors; ++c) {
            for (int col = 0; col < config::Board::kWidth; ++col) {
                for (int row = 0; row < 16; ++row) {
                    x += 0x9E3779B97F4A7C15ULL;
                    uint64_t z = x;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                    table[c][col][row] = z ^ (z >> 31);
                }
            }
        }
        initialized = true;
    }

    [[nodiscard]] static __forceinline uint64_t xorPuyo(Cell c, int col, int row) noexcept {
        assert(static_cast<int>(c) >= 0 && static_cast<int>(c) < config::Board::kNumColors);
        assert(col >= 0 && col < config::Board::kWidth);
        assert(row >= 0 && row < 16);
        return table[static_cast<int>(c)][col][row];
    }

    [[nodiscard]] static __forceinline uint64_t hashBoard(const Board& board) noexcept {
        init();
        uint64_t h = 0;
        for (int c = 0; c < config::Board::kNumColors; ++c) {
            const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
            uint64_t lo = bb.lo;
            while (lo) {
                const int idx = std::countr_zero(lo);
                lo &= (lo - 1);
                h ^= table[c][idx >> 4][idx & 15];
            }
            uint64_t hi = bb.hi & config::Board::kHiMask;
            while (hi) {
                const int idx = std::countr_zero(hi);
                hi &= (hi - 1);
                h ^= table[c][(idx >> 4) + 4][idx & 15];
            }
        }
        return h;
    }
};

} // namespace puyotan::search
