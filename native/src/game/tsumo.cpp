#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/config.hpp>
#include <cstdlib>

namespace puyotan {

Tsumo::Tsumo(int32_t seed) {
    setSeed(seed);
}

void Tsumo::setSeed(int32_t seed) {
    if (has_filled_ && initial_seed_ == seed) {
        return; // Skip 1000 loop regenerations if identically seeded
    }
    initial_seed_ = seed;
    seed_ = seed;
    has_filled_ = true;
    fillPool();
}

int Tsumo::nextInt() {
    seed_ ^= (seed_ << 13);
    seed_ ^= (seed_ >> 17);
    seed_ ^= (seed_ << 15);
    // JS does `this.y >>> 0`, which casts the bit pattern to uint32_t.
    // The JS `Math.abs()` is a no-op because `>>> 0` makes it positive.
    uint32_t r = static_cast<uint32_t>(seed_);
    return static_cast<int>(r & (config::Rule::kColors - 1));
}

Cell Tsumo::nextKind() {
    switch (nextInt()) {
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
