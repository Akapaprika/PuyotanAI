#include <immintrin.h>
#include <puyotan/core/gravity.hpp>

namespace puyotan {

namespace {
static constexpr uint32_t kColLaneMask = (1u << config::Board::kTotalRows) - 1;
} // anonymous namespace

uint32_t Gravity::execute(Board& board) noexcept {
    uint32_t fallen_mask = 0;

    for (int col = 0; col < config::Board::kWidth; ++col) {
        const uint32_t occ_lane = board.occupancy_.cols[col] & kColLaneMask;

        // 隙間がない列（下から詰まっている、または空列）は 1命令で即スキップ
        if (((occ_lane + 1) & occ_lane) == 0) {
            continue;
        }

        const int total_cnt    = static_cast<int>(_mm_popcnt_u32(occ_lane));
        const uint16_t new_occ = static_cast<uint16_t>((1u << total_cnt) - 1u);

        // -------------------------------------------------------------
        // 第1の穴（Gap 1）の位置と長さを 1 サイクルで特定
        // -------------------------------------------------------------
        const int h_start1   = static_cast<int>(_tzcnt_u32(~occ_lane));
        const uint32_t s1    = occ_lane >> h_start1;
        const int gap_len1   = static_cast<int>(_tzcnt_u32(s1));
        const uint32_t rem1  = s1 >> gap_len1;

        // 穴より下を保護するマスク
        const uint32_t mask1 = (1u << h_start1) - 1u;

        // -------------------------------------------------------------
        // パス 1: 最初の穴を一括スライド（90%以上のケースはここで完了）
        // -------------------------------------------------------------
        uint16_t lane[config::Board::kNumColors];
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint32_t orig = board.boards_[i].cols[col];
            lane[i] = static_cast<uint16_t>((orig & mask1) | ((orig >> gap_len1) & ~mask1));
        }

        // -------------------------------------------------------------
        // まだ上に別の穴があるか判定（2つ目の穴が存在する場合のみ実行）
        // -------------------------------------------------------------
        if (((rem1 + 1) & rem1) != 0) [[unlikely]] {
            // 第2の穴（Gap 2）を一括スライド
            const uint32_t occ_mid = (occ_lane & mask1) | ((occ_lane >> gap_len1) & ~mask1);
            const int h_start2     = static_cast<int>(_tzcnt_u32(~occ_mid));
            const uint32_t s2      = occ_mid >> h_start2;
            const int gap_len2     = static_cast<int>(_tzcnt_u32(s2));
            const uint32_t mask2   = (1u << h_start2) - 1u;

            #pragma unroll
            for (int i = 0; i < config::Board::kNumColors; ++i) {
                const uint32_t mid = lane[i];
                lane[i] = static_cast<uint16_t>((mid & mask2) | ((mid >> gap_len2) & ~mask2));
            }
        }

        // -------------------------------------------------------------
        // 盤面更新と差分フラグの記録
        // -------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint16_t compacted = lane[i];
            fallen_mask |= static_cast<uint32_t>(compacted != board.boards_[i].cols[col]) << i;
            board.boards_[i].cols[col] = compacted;
        }

        board.occupancy_.cols[col] = new_occ;
    }

    return fallen_mask;
}

} // namespace puyotan