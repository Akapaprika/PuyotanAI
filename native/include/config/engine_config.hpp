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

// ============================================================
// Score constants (Standard Puyo Puyo Rules)
// ============================================================
namespace Score {
    // Soft drop bonus: 1 point per row fallen.
    constexpr int kSoftDropBonusPerGrid = 1;

    // Chain bonus: 0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512
    constexpr int kChainBonuses[] = {
        0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512
    };
    constexpr int kChainBonusesSize = static_cast<int>(sizeof(kChainBonuses) / sizeof(kChainBonuses[0]));

    // Color bonus: count of colors erased in one step
    // 1 color: 0, 2 colors: 3, 3 colors: 6, 4 colors: 12, 5 colors: 24...
    constexpr int kColorBonuses[] = {
        0, 0, 3, 6, 12, 24
    };
    constexpr int kColorBonusesSize = static_cast<int>(sizeof(kColorBonuses) / sizeof(kColorBonuses[0]));

    // Group bonus: size of group erased (index is size - config::Rule::kConnectCount)
    // index 0 (size 4): 0, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11+: 10
    constexpr int kGroupBonuses[] = {
        0, 2, 3, 4, 5, 6, 7, 10
    };
    constexpr int kGroupBonusesSize = static_cast<int>(sizeof(kGroupBonuses) / sizeof(kGroupBonuses[0]));

    static constexpr int getChainBonus(int chain) {
        int idx = (chain < 1) ? 0 : (chain - 1);
        if (idx >= kChainBonusesSize) {
            return kChainBonuses[kChainBonusesSize - 1];
        }
        return kChainBonuses[idx];
    }

    static constexpr int getColorBonus(int count) {
        if (count < 1) {
            return 0;
        }
        if (count >= kColorBonusesSize) {
            return kColorBonuses[kColorBonusesSize - 1];
        }
        return kColorBonuses[count];
    }

    static constexpr int getGroupBonus(int size) {
        int idx = size - config::Rule::kConnectCount;
        if (idx < 0) {
            return 0;
        }
        if (idx >= kGroupBonusesSize) {
            return kGroupBonuses[kGroupBonusesSize - 1];
        }
        return kGroupBonuses[idx];
    }
}

} // namespace puyotan::config
