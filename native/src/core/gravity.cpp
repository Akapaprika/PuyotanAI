#include <immintrin.h>
#include <puyotan/core/gravity.hpp>

namespace puyotan {

namespace {

// 13bit盤面専用: 下位7bit(16KB) + 上位6bit(4KB) = 計20KB (L1Dキャッシュ 32KB に完全常駐！)
struct Pext13Table {
    uint8_t lo[128][128]; // 7bit mask x 7bit val = 16,384 bytes
    uint8_t hi[64][64];   // 6bit mask x 6bit val = 4,096 bytes

    Pext13Table() noexcept : lo{}, hi{} {
        // 下位 7bit テーブル初期化
        for (int mask = 0; mask < 128; ++mask) {
            for (int val = 0; val < 128; ++val) {
                uint8_t res = 0;
                int shift = 0;
                for (int b = 0; b < 7; ++b) {
                    if ((mask >> b) & 1) {
                        if ((val >> b) & 1) res |= static_cast<uint8_t>(1 << shift);
                        ++shift;
                    }
                }
                lo[mask][val] = res;
            }
        }
        // 上位 6bit テーブル初期化
        for (int mask = 0; mask < 64; ++mask) {
            for (int val = 0; val < 64; ++val) {
                uint8_t res = 0;
                int shift = 0;
                for (int b = 0; b < 6; ++b) {
                    if ((mask >> b) & 1) {
                        if ((val >> b) & 1) res |= static_cast<uint8_t>(1 << shift);
                        ++shift;
                    }
                }
                hi[mask][val] = res;
            }
        }
    }
};

static const Pext13Table kPext13;
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;

} // anonymous namespace

uint32_t Gravity::execute(Board& board) noexcept {
    uint32_t fallen_mask = 0;

    for (int col = 0; col < config::Board::kWidth; ++col) {
        const uint32_t occ_lane = board.occupancy_.cols[col] & kColLaneMask;

        // 隙間がない列（下から詰まっている、または空列）は即スキップ
        if (((occ_lane + 1) & occ_lane) == 0) {
            continue;
        }

        // 下位 7bit と 上位 6bit に分割
        const uint32_t mask_lo = occ_lane & 0x7Fu;
        const uint32_t mask_hi = (occ_lane >> 7) & 0x3Fu;
        
        const int shift_hi     = static_cast<int>(_mm_popcnt_u32(mask_lo));
        const int total_cnt    = static_cast<int>(_mm_popcnt_u32(occ_lane));
        const uint16_t new_occ = static_cast<uint16_t>((1u << total_cnt) - 1u);

        // 色ループ内で発生する mask_lo / mask_hi を使った暗黙のポインタ加算を完全に消去
        const uint8_t* p_lo = kPext13.lo[mask_lo];
        const uint8_t* p_hi = kPext13.hi[mask_hi];

        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint16_t lane = board.boards_[i].cols[col];
            
            const uint32_t val_lo = lane & 0x7Fu;
            
            const uint32_t val_hi = lane >> 7; 

            // ポインタからの直接参照により、L1キャッシュからの最短レイテンシを実現
            const uint16_t compacted = static_cast<uint16_t>(
                p_lo[val_lo] | (p_hi[val_hi] << shift_hi)
            );

            // 変化があった色のみフラグを立て、次回のスキャン対象を制限する
            fallen_mask |= static_cast<uint32_t>(compacted != lane) << i;
            board.boards_[i].cols[col] = compacted;
        }

        board.occupancy_.cols[col] = new_occ;
    }

    return fallen_mask;
}

} // namespace puyotan