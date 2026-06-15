#pragma once

#include <array>
#include <cstdint>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>

namespace puyotan {
/**
 * @class Tsumo
 * @brief Ring-buffered puyo piece generator with lazy chunk generation.
 *
 * Generates a deterministic sequence of puyo pairs based on a 32-bit seed.
 * Pool entries are generated on demand in chunks of kTsumoChunkSize to avoid
 * the cost of pre-computing all 1000 entries upfront.
 *
 * getSeed() (= ojama RNG seed) is computed via a precomputed GF(2)^32 jump
 * matrix, replacing the 2000-step sequential XORSHIFT with O(32) XOR ops.
 */
class Tsumo {
  public:
    explicit Tsumo(uint32_t seed = 1u) noexcept;

    /**
     * @brief Retrieves a PuyoPiece at the specified absolute sequence index.
     * @param index Sequence index (starts at 0, wraps at kTsumoPoolSize).
     * @return The axis and sub puyo colors.
     * @note O(1) normally. The [[unlikely]] branch fires at most once per
     *       kTsumoChunkSize calls, generating the next chunk in bulk.
     */
    inline PuyoPiece get(int32_t index) const noexcept {
        uint32_t idx = static_cast<uint32_t>(index);
        if (idx >= config::Rule::kTsumoPoolSize) [[unlikely]] {
            idx -= config::Rule::kTsumoPoolSize;
        }
        if (idx >= generated_count_) [[unlikely]] {
            expandTo(idx);
        }
        return pool_[idx];
    }

    void setSeed(uint32_t seed) noexcept;

    /// Returns the XORSHIFT state after all kTsumoPoolSize pairs have been
    /// consumed. Used by PuyotanMatch to seed the ojama RNG.
    /// Computed via jump matrix in setSeed() — O(32) ops, not O(2000).
    uint32_t getSeed() const noexcept { return ojama_seed_; }

  private:
    /// Generates pool entries up to the chunk boundary that covers target_idx.
    /// Called lazily from get() via [[unlikely]] branch.
    void expandTo(uint32_t target_idx) const noexcept;

    uint32_t ojama_seed_;  ///< XORSHIFT state after kTsumoPoolSize pairs

    mutable uint32_t rng_state_;        ///< Running state for lazy pool generation
    mutable uint32_t generated_count_;  ///< Number of valid entries in pool_
    mutable std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_;
};
} // namespace puyotan
