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
    explicit Tsumo(uint32_t seed = 0);

    inline PuyoPiece get(int index) const {
        return pool_[index];
    }
    void setSeed(uint32_t seed);
    uint32_t getSeed() const { return seed_; }

private:
    uint32_t seed_;
    uint32_t initial_seed_ = 0;
    bool has_filled_ = false;
    std::array<PuyoPiece, config::Rule::kTsumoPoolSize> pool_;

    int nextInt(int max);
    Cell nextKind();
    void fillPool();
};

} // namespace puyotan
