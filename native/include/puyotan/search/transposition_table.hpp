#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace puyotan::search {

/**
 * @class TranspositionTable
 * @brief 8バイト高密度エントリ置換表 (131,072 エントリ = 1.0 MB)
 * 
 * 盤面形状に対するポテンシャル得点はツモや手数に依存しないため、
 * 世代管理を行わず全ターン・全深さで無期限にキャッシュ再利用します。
 */
class TranspositionTable {
public:
    static constexpr std::size_t kSizePower = 17; // 2^17 = 131,072 entries
    static constexpr std::size_t kTableSize = 1ULL << kSizePower;
    static constexpr uint64_t    kMask      = kTableSize - 1;

    struct alignas(8) Entry {
        uint32_t key_hi;    // ハッシュ上位32bit
        int32_t  pot_score; // ポテンシャルスコア (-1: 未登録)
    };
    static_assert(sizeof(Entry) == 8, "Entry must be exactly 8 bytes");

    TranspositionTable() {
        table_.assign(kTableSize, Entry{0, -1});
    }

    void advanceGeneration() noexcept {
        // 盤面ポテンシャルは手数不変のため世代管理不要
    }

    void clear() noexcept {
        std::memset(table_.data(), 0xFF, table_.size() * sizeof(Entry)); // pot_score = -1
    }

    // 盤面が一致していればいつでも利用可能
    [[nodiscard]] __forceinline bool get(uint64_t key, int32_t& out_score) const noexcept {
        const Entry& e = table_[key & kMask];
        if (e.key_hi == static_cast<uint32_t>(key >> 32) && e.pot_score >= 0) {
            out_score = e.pot_score;
            return true;
        }
        return false;
    }

    __forceinline void put(uint64_t key, int32_t pot_score) noexcept {
        table_[key & kMask] = {static_cast<uint32_t>(key >> 32), pot_score};
    }

private:
    std::vector<Entry> table_; // 1.0 MB
};

inline thread_local TranspositionTable tl_tt;

} // namespace puyotan::search
