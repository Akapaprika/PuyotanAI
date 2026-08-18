#include <algorithm>
#include <bit>
#include <puyotan/common/config.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan {

namespace {

// ---------------------------------------------------------------------------
// GF(2)^32 Jump Matrix for XORSHIFT(13, 17, 15)
// ---------------------------------------------------------------------------
static constexpr std::array<uint32_t, 32> computeJumpMatrix() noexcept {
    auto xorshift_step = [](uint32_t s) constexpr noexcept -> uint32_t {
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        return s;
    };

    constexpr int kSteps = config::Rule::kTsumoPoolSize * 2; // 2000 steps
    std::array<uint32_t, 32> mat{};
    for (int i = 0; i < 32; ++i) {
        uint32_t s = 1u << i;
        for (int j = 0; j < kSteps; ++j) {
            s = xorshift_step(s);
        }
        mat[i] = s;
    }
    return mat;
}

static constexpr auto kJumpMatrix = computeJumpMatrix();

// 立っているビットのみを TZCNT/BLSR で高速走査
[[nodiscard]] static inline uint32_t applyJump(uint32_t state) noexcept {
    uint32_t result = 0;
    while (state) {
        const int i = std::countr_zero(state);
        result ^= kJumpMatrix[i];
        state &= (state - 1u);
    }
    return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Tsumo implementation
// ---------------------------------------------------------------------------

void Tsumo::setSeed(uint32_t seed) noexcept {
    const uint32_t s0 = seed + (seed == 0u);
    initial_seed_ = s0;
    rng_state_ = s0;
    generated_count_ = 0;
    ojama_seed_computed_ = false;

    // ★ 初期チャンクとして最初の 64 手分だけ高速生成 (高々 400 サイクル)
    // これにより秒間26万回のゲーム生成ベンチマークでの CPU 負荷を 1/15 に抑制
    expandTo(config::Rule::kTsumoChunkSize - 1u);
}

uint32_t Tsumo::getSeed() const noexcept {
    if (!ojama_seed_computed_) [[unlikely]] {
        ojama_seed_ = applyJump(initial_seed_);
        ojama_seed_computed_ = true;
    }
    return ojama_seed_;
}

void Tsumo::expandTo(uint32_t target_idx) const noexcept {
    constexpr uint32_t kChunk = config::Rule::kTsumoChunkSize;
    constexpr uint32_t kPoolSize = config::Rule::kTsumoPoolSize;
    constexpr uint32_t kColors = config::Rule::kColors;
    constexpr uint32_t color_mask = kColors - 1u;

    const uint32_t new_count =
        std::min((target_idx + kChunk) & ~(kChunk - 1u), kPoolSize);

    uint32_t s = rng_state_;

    for (uint32_t i = generated_count_; i < new_count; ++i) {
        // Axis puyo
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        const Cell c1 = static_cast<Cell>(s & color_mask);

        // Sub puyo
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        const Cell c2 = static_cast<Cell>(s & color_mask);

        // Precompute dirty flag for O(1) early-exit check in beam search
        const uint8_t dirty = static_cast<uint8_t>(
            (1u << static_cast<int>(c1)) | (1u << static_cast<int>(c2)));

        pool_[i] = {c1, c2, dirty, 0};
    }

    rng_state_ = s;
    generated_count_ = new_count;
}

} // namespace puyotan