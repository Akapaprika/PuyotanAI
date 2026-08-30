#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace puyotan::search {

/**
 * @class DepthDedupTable
 * @brief 同一深さ（同一手数）内での局面重複排除テーブル。
 * 
 * Zobrist ハッシュを用いて同一深さで既に探索・登録された盤面を高速に判定し、
 * 重複ノードを排除します。世代番号 (depth_gen) による管理を行うため、
 * 深さごとの初期化は O(1) で完了し、memset 等のオーバーヘッドがありません。
 */
class DepthDedupTable {
public:
    static constexpr size_t kSize = 131072; // 2^17 エントリ
    static constexpr size_t kMask = kSize - 1;

    struct Entry {
        uint64_t hash;
        uint32_t depth_gen;
    };

    DepthDedupTable() : table_(kSize, Entry{0, 0}) {}

    /// @brief 新しい深さに進む際に世代を進める (O(1))
    __forceinline void advanceDepth() noexcept {
        ++current_gen_;
        if (current_gen_ == 0) [[unlikely]] {
            // オーバーフロー時のリセット
            std::fill(table_.begin(), table_.end(), Entry{0, 0});
            current_gen_ = 1;
        }
    }

    /// @brief すでに同一深さで登録されていれば true (重複)、未登録なら登録して false (新規)
    [[nodiscard]] __forceinline bool checkAndInsert(uint64_t hash) noexcept {
        const size_t idx = hash & kMask;
        Entry& e = table_[idx];
        if (e.depth_gen == current_gen_ && e.hash == hash) {
            return true; // 同一深さでの重複！
        }
        e = {hash, current_gen_};
        return false; // 新規登録
    }

private:
    std::vector<Entry> table_; // 131,072 * 16 bytes = 2.0 MB
    uint32_t current_gen_ = 1;
};

inline thread_local DepthDedupTable tl_depth_dedup;

} // namespace puyotan::search
