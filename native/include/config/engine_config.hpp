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
    // BitBoard masks: Calculated from kTotalRows and kBitsPerCol
    //   Each column uses kBitsPerCol (16) bits.
    //   Only kTotalRows (14) bits are valid within each column lane.
    // --------------------------------------------------------
    
    // Generates a 14-bit mask: (1 << 14) - 1 = 0x3FFF
    static constexpr uint64_t kColMask = (1ULL << kTotalRows) - 1;

    // lo covers cols 0-3: [lane0 | lane1 | lane2 | lane3]
    constexpr uint64_t kLoMask = kColMask | 
                                 (kColMask << (1 * kBitsPerCol)) | 
                                 (kColMask << (2 * kBitsPerCol)) | 
                                 (kColMask << (3 * kBitsPerCol));

    // hi covers cols 4-5: [lane0 | lane1]
    constexpr uint64_t kHiMask = kColMask | 
                                 (kColMask << (1 * kBitsPerCol));

    // Mask isolating row 13 (spawn row) across all columns.
    constexpr uint64_t kLoSpawnMask = (1ULL << kSpawnRow) | 
                                      (1ULL << (kSpawnRow + 1 * kBitsPerCol)) | 
                                      (1ULL << (kSpawnRow + 2 * kBitsPerCol)) | 
                                      (1ULL << (kSpawnRow + 3 * kBitsPerCol));

    constexpr uint64_t kHiSpawnMask = (1ULL << kSpawnRow) | 
                                      (1ULL << (kSpawnRow + 1 * kBitsPerCol));

    // Mask isolating the full 16-bit lane of a column.
    static constexpr uint64_t kFullLaneMask = (1ULL << kBitsPerCol) - 1;

    // Mask isolating col 3 within lo (top 16-bit lane of lo).
    constexpr uint64_t kLoCol3Mask  = kFullLaneMask << (3 * kBitsPerCol);

    // Mask isolating col 4 within hi (bottom 16-bit lane of hi).
    constexpr uint64_t kHiCol4Mask  = kFullLaneMask;
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
