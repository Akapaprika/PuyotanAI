#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/config.hpp>
#include <cstdlib>

namespace puyotan {

Tsumo::Tsumo(uint32_t seed) {
    setSeed(seed);
}

void Tsumo::setSeed(uint32_t seed) {
    if (has_filled_ && initial_seed_ == seed) {
        return; // Skip 1000 loop regenerations if identically seeded
    }
    initial_seed_ = seed;
    seed_ = seed;
    has_filled_ = true;
    fillPool();
}

int Tsumo::nextInt(int max) {
    int32_t signed_y = static_cast<int32_t>(seed_);
    signed_y ^= (signed_y << 13);
    signed_y ^= (signed_y >> 17);
    signed_y ^= (signed_y << 15);
    seed_ = static_cast<uint32_t>(signed_y);
    
    uint32_t r = static_cast<uint32_t>(std::abs(signed_y));
    if (max == config::Rule::kColors) {
        return static_cast<int>(r & (config::Rule::kColors - 1));
    }
    return static_cast<int>(r % max);
}

Cell Tsumo::nextKind() {
    switch (nextInt(config::Rule::kColors)) {
        case 0:  return Cell::Red;
        case 1:  return Cell::Green;
        case 2:  return Cell::Blue;
        case 3:  return Cell::Yellow;
        default: return Cell::Empty;
    }
}

void Tsumo::fillPool() {
    for (int i = 0; i < config::Rule::kTsumoPoolSize; ++i) {
        pool_[i] = {nextKind(), nextKind()};
    }
}

} // namespace puyotan
