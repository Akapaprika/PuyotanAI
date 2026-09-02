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

        const int total_cnt    = static_cast<int>(_mm_popcnt_u32(occ));
        const uint16_t new_occ = static_cast<uint16_t>((1u << total_cnt) - 1u);

        // -------------------------------------------------------------
        // パス 1: board.boards_ から直接読み込みつつスライドして lane に格納
        // ★無駄な初期コピーを完全排除（95% はここだけで終了）
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

        // -------------------------------------------------------------
        // パス 2以降: 穴が2個以上ある場合のみ追加スライド（発生率 5% 未満）
        // -------------------------------------------------------------
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

        // 盤面更新と差分フラグ記録（1回のみ実行）
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