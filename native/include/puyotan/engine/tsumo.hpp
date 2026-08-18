#pragma once

#include <array>
#include <cstdint>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>

namespace puyotan {

class Tsumo {
  public:
    explicit Tsumo(uint32_t seed = 1u) noexcept {
        setSeed(seed);
    }

    [[nodiscard]] __forceinline PuyoPiece get(int32_t index) const noexcept {
        uint32_t idx = static_cast<uint32_t>(index);
        if (idx >= config::Rule::kTsumoPoolSize) [[unlikely]] {
            idx -= config::Rule::kTsumoPoolSize;
        }
        // 64手以上の超ロング探索時のみ稀に追加生成
        if (idx >= generated_count_) [[unlikely]] {
            expandTo(idx);
        }
        return pool_[idx];
    }

    void setSeed(uint32_t seed) noexcept;

    uint32_t getSeed() const noexcept;

  private:
    void expandTo(uint32_t target_idx) const noexcept;

    uint32_t initial_seed_ = 1u;
    mutable uint32_t rng_state_ = 1u;
    mutable uint32_t generated_count_ = 0;
    mutable uint32_t ojama_seed_ = 0;
    mutable bool ojama_seed_computed_ = false;

    // 64バイト境界に完全アライメント
    alignas(64) mutable std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_{};
};

} // namespace puyotan