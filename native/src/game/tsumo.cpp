#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/config.hpp>
#include <cstdlib>

namespace puyotan {

Tsumo::Tsumo(int32_t seed) {
    setSeed(seed);
}

void Tsumo::setSeed(int32_t seed) {
    seed_ = seed;
    generated_count_ = 0;
}

int Tsumo::nextInt() const {
    seed_ ^= (seed_ << 13);
    seed_ ^= (seed_ >> 17);
    seed_ ^= (seed_ << 15);
    // JS does `this.y >>> 0`, which casts the bit pattern to uint32_t.
    // The JS `Math.abs()` is a no-op because `>>> 0` makes it positive.
    uint32_t r = static_cast<uint32_t>(seed_);
    return static_cast<int>(r & (config::Rule::kColors - 1));
}

Cell Tsumo::nextKind() const {
    return static_cast<Cell>(nextInt());
}

void Tsumo::generateUpTo(int target_index) const {
    while (generated_count_ <= target_index && generated_count_ < config::Rule::kTsumoPoolSize) {
        pool_[generated_count_] = {nextKind(), nextKind()};
        ++generated_count_;
    }
}

} // namespace puyotan
