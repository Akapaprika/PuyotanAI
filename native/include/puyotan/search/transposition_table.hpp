#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace puyotan::search {

class TranspositionTable {
public:
    static constexpr std::size_t kSizePower = 17; // 2^17 = 131,072 entries
    static constexpr std::size_t kTableSize = 1ULL << kSizePower;
    static constexpr uint64_t    kMask      = kTableSize - 1;

    struct alignas(16) Entry {
        uint64_t key;
        int32_t  pot_score;
        uint32_t gen;
    };
    static_assert(sizeof(Entry) == 16, "Entry must be exactly 16 bytes");

    TranspositionTable() {
        table_.assign(kTableSize, Entry{0, -1, 0});
    }

    void advanceGeneration() noexcept {
        if (++current_gen_ == 0) [[unlikely]] {
            std::memset(table_.data(), 0, table_.size() * sizeof(Entry));
            current_gen_ = 1;
        }
    }

    void clear() noexcept {
        std::memset(table_.data(), 0, table_.size() * sizeof(Entry));
        current_gen_ = 1;
    }

    // 盤面が一致していれば世代によらずそのまま利用可能
    [[nodiscard]] __forceinline bool get(uint64_t key, int32_t& out_score) const noexcept {
        const Entry& e = table_[key & kMask];
        if (e.key == key && e.pot_score >= 0) {
            out_score = e.pot_score;
            return true;
        }
        return false;
    }

    __forceinline void put(uint64_t key, int32_t pot_score) noexcept {
        table_[key & kMask] = {key, pot_score, current_gen_};
    }

private:
    std::vector<Entry> table_; // 2.0 MB
    uint32_t current_gen_ = 1;
};

inline thread_local TranspositionTable tl_tt;

} // namespace puyotan::search
