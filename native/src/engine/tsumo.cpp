#include <algorithm>
#include <puyotan/common/config.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan {

namespace {

static constexpr std::array<uint32_t, 32> computeJumpMatrix() noexcept {
    auto xorshift_step = [](uint32_t s) constexpr noexcept -> uint32_t {
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        return s;
    };

    constexpr int kSteps = config::Rule::kTsumoPoolSize * 2;
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

[[nodiscard]] static inline uint32_t applyJump(uint32_t state) noexcept {
    uint32_t result = 0;
    for (int i = 0; i < 32; ++i) {
        if ((state >> i) & 1u)
            result ^= kJumpMatrix[i];
    }
    return result;
}

} // anonymous namespace

TsumoSequence::TsumoSequence(uint32_t seed) noexcept {
    setSeed(seed);
}

void TsumoSequence::setSeed(uint32_t seed) noexcept {
    const uint32_t s0 = seed + (seed == 0u);
    initial_seed_ = s0;
    rng_state_ = s0;
    generated_count_ = 0;
    ojama_seed_computed_ = false;

    // 対局・探索の初期化時点では、最初の 1 チャンク（64手）のみを高速事前生成
    expandTo(63);
}

uint32_t TsumoSequence::getOjamaSeed() const noexcept {
    if (!ojama_seed_computed_) {
        ojama_seed_ = applyJump(initial_seed_);
        ojama_seed_computed_ = true;
    }
    return ojama_seed_;
}

void TsumoSequence::expandTo(uint32_t target_idx) const noexcept {
    std::lock_guard<std::mutex> lock(expand_mutex_);

    // 二重チェック（ロック獲得中に別スレッドが生成完了している場合のショートカット）
    if (target_idx < generated_count_) {
        return;
    }

    constexpr uint32_t kChunk = config::Rule::kTsumoChunkSize; // 64
    constexpr uint32_t kPoolSize = config::Rule::kTsumoPoolSize; // 1000
    constexpr uint32_t kColors = config::Rule::kColors;
    constexpr uint32_t color_mask = kColors - 1u;

    // 64手単位（Chunk）に切り上げて拡張
    const uint32_t new_count = std::min((target_idx + kChunk) & ~(kChunk - 1u), kPoolSize);

    uint32_t s = rng_state_;
    for (uint32_t i = generated_count_; i < new_count; ++i) {
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        const Cell c1 = static_cast<Cell>(s & color_mask);

        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        const Cell c2 = static_cast<Cell>(s & color_mask);

        const uint8_t dirty = static_cast<uint8_t>(
            (1u << static_cast<int>(c1)) | (1u << static_cast<int>(c2)));

        pool_[i] = {c1, c2, dirty, 0};
    }

    rng_state_ = s;
    generated_count_ = new_count;
}

} // namespace puyotan