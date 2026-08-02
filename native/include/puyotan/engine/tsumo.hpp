#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>

namespace puyotan {

// 64手ごとの Lazy Chunk 生成を行う共有ツモプール (約4KB)
class TsumoSequence {
  public:
    explicit TsumoSequence(uint32_t seed = 1u) noexcept;

    // 不用意な4KBコピーを防ぐためコピーは削除（参照・ポインタ共有を強制）
    TsumoSequence(const TsumoSequence&) = delete;
    TsumoSequence& operator=(const TsumoSequence&) = delete;

    void setSeed(uint32_t seed) noexcept;

    // インデックス指定でツモを取得（64手以内なら0オーバーヘッドで直参照）
    [[nodiscard]] inline PuyoPiece get(int32_t index) const noexcept {
        uint32_t idx = static_cast<uint32_t>(index);
        if (idx >= config::Rule::kTsumoPoolSize) [[unlikely]] {
            idx %= config::Rule::kTsumoPoolSize;
        }
        if (idx >= generated_count_) [[unlikely]] {
            expandTo(idx);
        }
        return pool_[idx];
    }

    // 遅延評価・事前計算されたおじゃまシードを取得
    uint32_t getOjamaSeed() const noexcept;

  private:

    void expandTo(uint32_t target_idx) const noexcept;

    uint32_t initial_seed_ = 1u;
    mutable uint32_t rng_state_ = 1u;
    mutable uint32_t generated_count_ = 0;
    mutable uint32_t ojama_seed_ = 0;
    mutable bool ojama_seed_computed_ = false;

    mutable std::mutex expand_mutex_; // マルチスレッド安全な追加生成用
    mutable std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_{};
};

// 探索ノード用超軽量カーソル (16バイト)
struct TsumoCursor {
    const TsumoSequence* sequence = nullptr;
    uint16_t position = 0;

    [[nodiscard]] __forceinline PuyoPiece get(int offset = 0) const noexcept {
        assert(sequence != nullptr);
        return sequence->get(position + offset);
    }

    __forceinline void advance(int count = 1) noexcept {
        position = static_cast<uint16_t>(position + count);
    }
};

} // namespace puyotan