#pragma once

#include <array>
#include <cstdint>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>

namespace puyotan {

/**
 * @class Tsumo
 * @brief Thread-safe (read-only) ring-buffered puyo piece generator.
 * 
 * Generates a deterministic sequence of puyo pairs based on a 32-bit seed.
 * Pieces are cached in a fixed-size pool to minimize RNG overhead during simulation.
 */
class Tsumo {
public:
    /**
     * @brief Retrieves a PuyoPiece at the specified absolute sequence index.
     * @param index sequence index (starts at 0).
     * @return The axis and sub puyo colors.
     * @note Performance: O(1) ring-buffer access.
     */
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
