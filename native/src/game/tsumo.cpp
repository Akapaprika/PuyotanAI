#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/config.hpp>
#include <cstdlib>

namespace puyotan {

Tsumo::Tsumo(uint32_t seed) noexcept {
    setSeed(seed);
}

void Tsumo::setSeed(uint32_t seed) noexcept {
    // Replace 0 with 1 mathematically to avoid conditional move (CMOV) overhead
    seed_ = seed + (seed == 0); // XORSHIFT requires non-zero seed
    generated_count_ = 0;
    generateMore(); // Initial chunk
}

void Tsumo::generateMore() noexcept {
    // Optimization: Since ChunkSize (64) is a power of 2 and a factor of PoolSize (256),
    // and generated_count_ always starts at 0 and increments by 64, 
    // we never cross the wrap-around boundary within a single generateMore() call.
    static_assert((config::Rule::kTsumoPoolSize % config::Rule::kTsumoChunkSize) == 0,
                  "ChunkSize must be a factor of PoolSize for fast contiguous writes");

    uint32_t s = seed_;
    const size_t start_idx = static_cast<size_t>(generated_count_) & (config::Rule::kTsumoPoolSize - 1);
    PuyoPiece* __restrict p = &pool_[start_idx];
    const uint32_t color_mask = config::Rule::kColors - 1;

    for (int i = 0; i < config::Rule::kTsumoChunkSize; ++i) {
        // Generate Axis Puyo (XORSHIFT 13, 17, 15)
        s ^= (s << 13);
        s ^= (s >> 17);
        s ^= (s << 15);
        Cell c1 = static_cast<Cell>(s & color_mask);

        // Generate Sub Puyo
        s ^= (s << 13);
        s ^= (s >> 17);
        s ^= (s << 15);
        Cell c2 = static_cast<Cell>(s & color_mask);

        // Direct contiguous write without inner-loop masking
        p[i] = {c1, c2};
    }

    seed_ = s;
    generated_count_ += config::Rule::kTsumoChunkSize;
}

} // namespace puyotan
