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
    explicit Tsumo(int32_t seed = 0);

    inline PuyoPiece get(int index) const {
        if (index >= generated_count_) {
            generateMore();
        }
        // Ring buffer access using bitmask (size 512 -> mask 0x1FF)
        return pool_[index & (config::Rule::kTsumoPoolSize - 1)];
    }
    void setSeed(int32_t seed);
    int32_t getSeed() const { return seed_; }

private:
    mutable int32_t seed_;
    mutable int generated_count_ = 0;
    mutable std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_;

    int nextInt() const;
    Cell nextKind() const;
    void generateMore() const;
};

} // namespace puyotan
