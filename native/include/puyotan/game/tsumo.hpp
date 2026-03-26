#pragma once

#include <array>
#include <cstdint>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>

namespace puyotan {

/**
 * Tsumo
 *   Generates and maintains a pool of puyo pairs.
 */
class Tsumo {
public:
    explicit Tsumo(uint32_t seed = 1u) noexcept;

    inline PuyoPiece get(int32_t index) noexcept {
        if (index >= generated_count_) {
            generateMore();
        }
        // Ring buffer access using bitmask
        return pool_[static_cast<std::size_t>(index) & (config::Rule::kTsumoPoolSize - 1)];
    }
    void setSeed(uint32_t seed) noexcept;
    uint32_t getSeed() const noexcept { return seed_; }

private:
    uint32_t seed_;
    int32_t generated_count_ = 0;
    std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_;

    static_assert((config::Rule::kTsumoPoolSize & (config::Rule::kTsumoPoolSize - 1)) == 0, 
                  "TsumoPoolSize must be a power of 2 for fast bitmask indexing");

    void generateMore() noexcept;
};

} // namespace puyotan
