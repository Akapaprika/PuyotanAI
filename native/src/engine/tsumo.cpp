#include <cstdlib>
#include <puyotan/common/config.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan {
Tsumo::Tsumo(uint32_t seed) noexcept {
    setSeed(seed);
}

void Tsumo::setSeed(uint32_t seed) noexcept {
    // seed can be any positive uint32. XORSHIFT requires non-zero.
    seed_ = seed + (seed == 0);
    uint32_t s = seed_;
    const uint32_t color_mask = config::Rule::kColors - 1;

    for (int i = 0; i < config::Rule::kTsumoPoolSize; ++i) {
        // Generate Axis Puyo (XORSHIFT 13, 17, 15)
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        Cell c1 = static_cast<Cell>(s & color_mask);

        // Generate Sub Puyo
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        Cell c2 = static_cast<Cell>(s & color_mask);

        // Precompute dirty flag for O(1) retrieval during simulation
        uint8_t dirty = static_cast<uint8_t>((1u << static_cast<int>(c1)) | (1u << static_cast<int>(c2)));

        pool_[i] = {c1, c2, dirty, 0};
    }
    seed_ = s;
}
} // namespace puyotan
