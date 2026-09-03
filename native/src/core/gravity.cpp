#include <immintrin.h>
#include <puyotan/core/gravity.hpp>

namespace puyotan {

uint32_t Gravity::execute(Board& board) noexcept {
    uint32_t fallen_mask = 0;

    for (int col = 0; col < config::Board::kWidth; ++col) {
        uint32_t occ = board.occupancy_.cols[col];

        // 隙間がない列（下から詰まっている、または空列）は 1命令で即スキップ
        if (((occ + 1) & occ) == 0) {
            continue;
        }

        // -------------------------------------------------------------
        // ★ total_cnt も new_occ も丸ごと消滅！
        // 穴を詰めるクリティカルパス（h1, g1, m1）だけに集中
        // -------------------------------------------------------------
        const int h1 = static_cast<int>(_tzcnt_u32(~occ));
        const uint32_t s1 = occ >> h1;
        const int g1 = static_cast<int>(_tzcnt_u32(s1));
        const uint32_t m1 = (1u << h1) - 1u;

        uint16_t lane[config::Board::kNumColors];
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint32_t orig = board.boards_[i].cols[col];
            lane[i] = static_cast<uint16_t>((orig & m1) | ((orig >> g1) & ~m1));
        }

        occ = (occ & m1) | ((occ >> g1) & ~m1);

        while (((occ + 1) & occ) != 0) {
            const int h = static_cast<int>(_tzcnt_u32(~occ));
            const uint32_t s = occ >> h;
            const int g = static_cast<int>(_tzcnt_u32(s));
            const uint32_t m = (1u << h) - 1u;

            #pragma unroll
            for (int i = 0; i < config::Board::kNumColors; ++i) {
                const uint32_t mid = lane[i];
                lane[i] = static_cast<uint16_t>((mid & m) | ((mid >> g) & ~m));
            }

            occ = (occ & m) | ((occ >> g) & ~m);
        }

        // 盤面更新
        #pragma unroll
        for (int i = 0; i < config::Board::kNumColors; ++i) {
            const uint16_t compacted = lane[i];
            fallen_mask |= static_cast<uint32_t>(compacted != board.boards_[i].cols[col]) << i;
            board.boards_[i].cols[col] = compacted;
        }

        // ★ 詰め終わった occ をそのまま書き込むだけ（計算コスト 0）
        board.occupancy_.cols[col] = static_cast<uint16_t>(occ);
    }

    return fallen_mask;
}

} // namespace puyotan