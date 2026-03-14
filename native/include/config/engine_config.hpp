#pragma once

#include <cstdint>

namespace puyotan::config {

// ============================================================
// Board dimensions (Puyotan β spec)
//   - visible field: 6 cols × 13 rows
//   - row 13 (0-indexed) is the invisible "spawn" row used by
//     setAndFall(); it is cleared after every gravity pass.
// ============================================================
namespace Board {
    constexpr int kWidth        = 6;   // number of columns
    constexpr int kHeight       = 13;  // visible rows (1-indexed: rows 1-13)
    constexpr int kSpawnRow     = 13;  // 0-indexed invisible 14th row
    constexpr int kTotalRows    = 14;  // kHeight + 1 (spawn row)
    constexpr int kBitsPerCol   = 16;  // bits allocated per column in the BitBoard
    constexpr int kColsInLo     = 4;   // columns 0-3 packed into lo (uint64_t)
    constexpr int kColsInHi     = 2;   // columns 4-5 packed into hi (uint64_t)
    constexpr int kNumColors    = 5;   // Red, Green, Blue, Yellow, Ojama

    // --------------------------------------------------------
    // BitBoard masks: 14 rows × 16 bits/col
    //   lo covers cols 0-3: each 16-bit lane -> 14 bits valid
    //     0x3FFF = 0b0011_1111_1111_1111 (bits 0-13)
    //   hi covers cols 4-5: lower two 16-bit lanes only
    // --------------------------------------------------------
    constexpr uint64_t kLoMask  = 0x3FFF'3FFF'3FFF'3FFFull; // cols 0-3
    constexpr uint64_t kHiMask  = 0x0000'0000'3FFF'3FFFull; // cols 4-5

    // Mask isolating row 13 (spawn row) across all columns.
    // Each column's bit-13 is at position: col*16 + 13.
    //   lo: cols 0-3 -> bits 13, 29, 45, 61
    //   hi: cols 4-5 -> bits 13, 29 (within hi's own 64-bit word)
    constexpr uint64_t kLoSpawnMask =
        (1ULL << 13) | (1ULL << 29) | (1ULL << 45) | (1ULL << 61);
    constexpr uint64_t kHiSpawnMask =
        (1ULL << 13) | (1ULL << 29);

    // Mask isolating col 3 within lo (top 16-bit lane of lo).
    // Used when transferring col 3 → col 4 in shift_right.
    constexpr uint64_t kLoCol3Mask  = 0xFFFF'0000'0000'0000ull;

    // Mask isolating col 4 within hi (bottom 16-bit lane of hi).
    // Used when transferring col 4 → col 3 in shift_left.
    constexpr uint64_t kHiCol4Mask  = 0x0000'0000'0000'FFFFull;
}

// ============================================================
// Rule constants
// ============================================================
namespace Rule {
    constexpr int kConnectCount = 4;  // minimum group size to fire
    constexpr int kColors       = 4;  // number of normal puyo colors
    constexpr int kPuyosPerPiece = 2; // number of puyos in each falling piece (tsumo)
    constexpr int kTsumoPoolSize = 1000; // size of pre-generated tsumo pool
    constexpr int kDeathCol     = 2;  // column index for death check (1-indexed: 3)
    constexpr int kDeathRow     = 11; // row index for death check (1-indexed: 12)
}

} // namespace puyotan::config
