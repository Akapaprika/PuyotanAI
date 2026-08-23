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

    // ★ ファストパス: インデックスを参照で受け取り、ラップアラウンド時は getSlow が書き戻す
    // 呼び出し元は 1000 ループの仕様を意識しなくてよい
    [[nodiscard]] __forceinline PuyoPiece get(int32_t& index) const noexcept {
        const uint32_t idx = static_cast<uint32_t>(index);
        if (idx >= generated_count_) [[unlikely]] {
            return getSlow(index);
        }
        return pool_[idx];
    }

    void setSeed(uint32_t seed) noexcept;
    uint32_t getSeed() const noexcept;

  private:
    void expandTo(uint32_t target_idx) const noexcept;

    // ★ スローパス: LTOによるインライン化を強制拒否し、ファストパスの肥大化を防ぐ
    // int32_t& を受け取り、ラップアラウンド後の値を書き戻す（lvalue ref版 get() から呼ばれる）
#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    PuyoPiece getSlow(int32_t& index) const noexcept;

    uint32_t initial_seed_ = 1u;
    mutable uint32_t rng_state_ = 1u;
    mutable uint32_t generated_count_ = 0;
    mutable uint32_t ojama_seed_ = 0;
    mutable bool ojama_seed_computed_ = false;

    alignas(64) mutable std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_{};
};

} // namespace puyotan