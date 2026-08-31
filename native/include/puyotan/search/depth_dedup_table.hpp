#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace puyotan::search {

/**
 * @class DepthDedupTable
 * @brief 同一深さ内での局面重複排除テーブル（64bit 単一命令判定・8バイト高密度版）
 * 
 * 131,072 エントリ × 8 バイト = 1.0 MB (AMD 3020e の 4MB L3 キャッシュに完全常駐)
 * - 上位 48bit: Zobrist ハッシュタグ
 * - 下位 16bit: depth_gen (世代番号)
 * 添字(17bit) + タグ(48bit) = 65bit 照合により、64bitハッシュの欠損ゼロ（擬陽性ゼロ）
 */
class DepthDedupTable {
public:
    static constexpr size_t kSize = 131072; // 2^17 エントリ
    static constexpr size_t kMask = kSize - 1;

    DepthDedupTable() : table_(kSize, 0) {}

    /// @brief 新しい深さに進む際に世代を進める (O(1))
    __forceinline void advanceDepth() noexcept {
        ++current_gen_;
        if (current_gen_ == 0) [[unlikely]] {
            // 約 2,180 ターンに 1 回の一括リセット (1.0 MB)
            std::memset(table_.data(), 0, table_.size() * sizeof(uint64_t));
            current_gen_ = 1;
        }
    }

    /// @brief すでに同一深さで登録されていれば true (重複)、未登録なら登録して false (新規)
    [[nodiscard]] __forceinline bool checkAndInsert(uint64_t hash) noexcept {
        const size_t idx = hash & kMask;
        // 上位48bitのハッシュと下位16bitの世代番号を 1 つの 64bit 整数にパック
        const uint64_t target = (hash & 0xFFFFFFFFFFFF0000ULL) | static_cast<uint64_t>(current_gen_);

        uint64_t& entry = table_[idx];
        if (entry == target) {
            return true; // 1 命令の 64bit cmp で完全一致判定
        }
        entry = target; // 1 命令の 64bit mov で登録
        return false;
    }

private:
    std::vector<uint64_t> table_; // 1.0 MB
    uint16_t current_gen_ = 1;
};

inline thread_local DepthDedupTable tl_depth_dedup;

} // namespace puyotan::search
