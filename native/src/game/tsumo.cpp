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
    generateMore(); // Initial 128
}

int Tsumo::nextInt() const {
    seed_ ^= (seed_ << 13);
    seed_ ^= (seed_ >> 17);
    seed_ ^= (seed_ << 15);
    uint32_t r = static_cast<uint32_t>(seed_);
    return static_cast<int>(r & (config::Rule::kColors - 1));
}

Cell Tsumo::nextKind() const {
    return static_cast<Cell>(nextInt());
}

void Tsumo::generateMore() const {
    // We can generate beyond kTsumoPoolSize because pool_ is used as a ring buffer.
    // However, generated_count_ represents the absolute count of pieces generated.
    int next_limit = generated_count_ + config::Rule::kTsumoChunkSize;

    while (generated_count_ < next_limit) {
        // Use mask for array index
        pool_[generated_count_ & (config::Rule::kTsumoPoolSize - 1)] = {nextKind(), nextKind()};
        ++generated_count_;
    }
}

} // namespace puyotan
