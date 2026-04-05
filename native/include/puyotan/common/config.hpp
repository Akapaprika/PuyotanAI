#pragma once

#include <cstdint>

namespace puyotan::config {

// ============================================================
// Board dimensions (Puyotan beta spec)
//   - visible field: 6 cols x 13 rows
//   - row 13 (0-indexed) is the invisible "spawn" row used by
//     setAndFall(); it is cleared after every gravity pass.
// ============================================================
namespace Board {
constexpr int kWidth = 6;               // number of columns
constexpr int kSpawnRow = 13;           // 0-indexed invisible 14th row
constexpr int kHeight = 13;             // visible rows (0-12)
constexpr int kTotalRows = kHeight + 2; // visible + spawn + sub-puyo (up rotation)
constexpr int kObsHeight = 14;          // Observation tensor height (13 rows board + 1 row metadata)
constexpr int kBitsPerCol = 16;         // bits allocated per column in the BitBoard
constexpr int kColsInLo = 4;            // columns 0-3 packed into lo (uint64_t)
constexpr int kColsInHi = 2;            // columns 4-5 packed into hi (uint64_t)
constexpr int kNumColors = 5;           // Red, Green, Blue, Yellow, Ojama

// --------------------------------------------------------
// BitBoard masks: Calculated from kTotalRows and kBitsPerCol
//   In the new 128-bit unified BitBoard:
//   Lo (bits 0-63):   Cols 0, 1, 2, 3
//   Hi (bits 64-127): Cols 4, 5
// --------------------------------------------------------

// Mask for a single 16-bit column lane (bits 0-14 used by visible + spawn rows)
static constexpr uint64_t kColMask = (1ULL << kTotalRows) - 1;

// Mask for only visible rows (0-12) in a single lane
static constexpr uint32_t kVisibleColMask = (1U << kHeight) - 1;

// lo covers cols 0-3: [lane0 | lane1 | lane2 | lane3]
constexpr uint64_t kLoMask = kColMask |
                             (kColMask << (1 * kBitsPerCol)) |
                             (kColMask << (2 * kBitsPerCol)) |
                             (kColMask << (3 * kBitsPerCol));

constexpr uint64_t kHiMask = kColMask |
                             (kColMask << (1 * kBitsPerCol));

// --------------------------------------------------------
// Chainable mask: Ghost row (13th row, index 12) doesn't pop.
// So only rows 0-11 (12 rows) are chainable.
// --------------------------------------------------------
constexpr int kChainableRows = 12; // Rows 0-11
static constexpr uint64_t kChainableColMask = (1ULL << kChainableRows) - 1;

constexpr uint64_t kChainableLoMask = kChainableColMask |
                                      (kChainableColMask << (1 * kBitsPerCol)) |
                                      (kChainableColMask << (2 * kBitsPerCol)) |
                                      (kChainableColMask << (3 * kBitsPerCol));

constexpr uint64_t kChainableHiMask = kChainableColMask |
                                      (kChainableColMask << (1 * kBitsPerCol));

// Mask for a full 16-bit lane (bits 0-15)
static constexpr uint32_t kFullLaneMask = (1U << kBitsPerCol) - 1;
} // namespace Board

// ============================================================
// Rule constants
// ============================================================
namespace Rule {
constexpr int kNumPlayers = 2;   // number of players in a standard match
constexpr int kConnectCount = 4; // minimum group size to fire
constexpr int kColors = 4;       // number of normal puyo colors
static_assert((kColors & (kColors - 1)) == 0, "kColors must be power of 2 for bitmask optimization in Tsumo");
constexpr int kPuyosPerPiece = 2;                   // number of puyos in each falling piece (tsumo)
constexpr int kTsumoPoolSize = 256;                 // Ring buffer size (2^8)
constexpr int kTsumoChunkSize = 64;                 // increment size (2^6)
constexpr int kDeathCol = 2;                        // column index for death check (1-indexed: 3)
constexpr int kDeathRow = 11;                       // row index for death check (1-indexed: 12)
constexpr int kMaxOjamaPerFall = Board::kWidth * 5; // standard: 30

// Max possible groups = floor(TotalCells / kConnectCount)
// 6 * 15 / 4 = 22.5. We use 24 for alignment/headroom.
constexpr int kMaxErasureGroups = (config::Board::kWidth * config::Board::kTotalRows) / kConnectCount + 2;
} // namespace Rule

// ============================================================
// Score constants (Standard Puyo Puyo Rules)
// ============================================================
namespace Score {
// Soft drop bonus: 1 point per row fallen.
constexpr int kSoftDropBonusPerGrid = 1;

// Chain bonus: 0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512
constexpr int kChainBonuses[] = {
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512};
constexpr int kChainBonusesSize = static_cast<int>(sizeof(kChainBonuses) / sizeof(kChainBonuses[0]));

// Color bonus: count of colors erased in one step
constexpr int kColorBonuses[] = {
    0, 0, 3, 6, 12, 24};
constexpr int kColorBonusesSize = static_cast<int>(sizeof(kColorBonuses) / sizeof(kColorBonuses[0]));

// Group bonus: size of group erased (index is size - config::Rule::kConnectCount)
constexpr int kGroupBonuses[] = {
    0, 2, 3, 4, 5, 6, 7, 10};
constexpr int kGroupBonusesSize = static_cast<int>(sizeof(kGroupBonuses) / sizeof(kGroupBonuses[0]));

// Amount of score required to generate one Ojama Puyo
constexpr int kTargetScore = 70;

// All Clear bonus: equivalent to 30 Ojama Puyos (6 columns * 5 rows * 70 score)
constexpr int kAllClearBonus = config::Board::kWidth * 5 * kTargetScore;
} // namespace Score

} // namespace puyotan::config
