#include <immintrin.h>
#include <puyotan/core/gravity.hpp>

namespace puyotan {

namespace {

// 256x256 の 8-bit PEXT テーブル (64 KB, プログラム起動時に一瞬で初期化)
struct Pext8Table {
    uint8_t data[256][256];

    Pext8Table() noexcept : data{} {
        for (int mask = 0; mask < 256; ++mask) {
            for (int val = 0; val < 256; ++val) {
                uint8_t res = 0;
                int shift = 0;
                for (int b = 0; b < 8; ++b) {
                    if ((mask >> b) & 1) {
                        if ((val >> b) & 1) {
                            res |= static_cast<uint8_t>(1 << shift);
                        }
                        ++shift;
                    }
                }
                data[mask][val] = res;
            }
        }
    }
};

static const Pext8Table kPext8;
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

} // anonymous namespace

// -----------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------

uint32_t Gravity::execute(Board& board) noexcept {
    uint32_t fallen_mask = 0;

    for (int col = 0; col < config::Board::kWidth; ++col) {
        const uint32_t occ_lane = board.occupancy_.cols[col] & kColLaneMask;

        // 隙間がない列（下から詰まっている、または空列）は即スキップ
        if (((occ_lane + 1) & occ_lane) == 0) {
            continue;
        }

        // 8bit 分割用のマスクと上位シフト量を事前計算
        const uint32_t mask_lo = occ_lane & 0xFFu;
        const uint32_t mask_hi = (occ_lane >> 8) & 0xFFu;
        const int shift_hi     = static_cast<int>(_mm_popcnt_u32(mask_lo));
        const int total_cnt    = shift_hi + static_cast<int>(_mm_popcnt_u32(mask_hi));
        const uint16_t new_occ = static_cast<uint16_t>((1u << total_cnt) - 1u);

        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint16_t lane = board.boards_[i].cols[col];
            if (lane == 0) continue;

            const uint32_t val_lo = lane & 0xFFu;
            const uint32_t val_hi = (lane >> 8) & 0xFFu;

            const uint16_t compacted = static_cast<uint16_t>(
                kPext8.data[mask_lo][val_lo] | (kPext8.data[mask_hi][val_hi] << shift_hi)
            );

            // ブランチレス: 無条件代入 + フラグ更新
            fallen_mask |= static_cast<uint32_t>(compacted != lane) << i;
            board.boards_[i].cols[col] = compacted;
        }

        board.occupancy_.cols[col] = new_occ;
    }

    return fallen_mask;
}

} // namespace puyotan