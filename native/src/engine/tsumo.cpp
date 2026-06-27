#include <algorithm>
#include <puyotan/common/config.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan {

namespace {

// ---------------------------------------------------------------------------
// GF(2)^32 Jump Matrix for XORSHIFT(13, 17, 15)
// ---------------------------------------------------------------------------
// XORSHIFT(13,17,15) is a linear transformation T over GF(2)^32.
// We precompute the matrix for T^(kTsumoPoolSize * 2) — i.e., the state
// after advancing through all 1000 pairs (2000 XORSHIFT steps).
//
// jump_matrix[i] = T^2000(1 << i)
//
// Applying the matrix to an arbitrary state takes only 32 XOR operations:
//   result = XOR of jump_matrix[i] for every set bit i in state
//
// This replaces the 2000-step sequential XORSHIFT loop in getSeed() with
// an O(32) operation, achieving ~60x speedup for that computation.
// ---------------------------------------------------------------------------

static constexpr std::array<uint32_t, 32> computeJumpMatrix() noexcept {
    // One XORSHIFT(13,17,15) step (C++23: arithmetic right shift is defined)
    auto xorshift_step = [](uint32_t s) constexpr noexcept -> uint32_t {
        s ^= (s << 13);
        s ^= static_cast<uint32_t>(static_cast<int32_t>(s) >> 17);
        s ^= (s << 15);
        return s;
    };

    // Column i of T^N = T^N applied to the basis vector (1 << i)
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

// Precomputed at compile time — zero runtime cost.
static constexpr auto kJumpMatrix = computeJumpMatrix();

// Apply the jump matrix: computes T^2000(state) in 32 XOR operations.
[[nodiscard]] static inline uint32_t applyJump(uint32_t state) noexcept {
    uint32_t result = 0;
    for (int i = 0; i < 32; ++i) {
        if ((state >> i) & 1u)
            result ^= kJumpMatrix[i];
    }
    return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Tsumo implementation
// ---------------------------------------------------------------------------

Tsumo::Tsumo(uint32_t seed) noexcept {
    setSeed(seed);
}

void Tsumo::setSeed(uint32_t seed) noexcept {
    const uint32_t s0 = seed + (seed == 0u);
    rng_state_ = s0;
    initial_seed_ = s0;
    generated_count_ = 0;
    ojama_seed_computed_ = false;

    expandTo(63);
}

uint32_t Tsumo::getSeed() const noexcept {
    if (!ojama_seed_computed_) {
        ojama_seed_ = applyJump(initial_seed_);
        ojama_seed_computed_ = true;
    }
    return ojama_seed_;
}

void Tsumo::expandTo(uint32_t target_idx) const noexcept {
    constexpr uint32_t kChunk = config::Rule::kTsumoChunkSize;
    constexpr uint32_t kPoolSize = config::Rule::kTsumoPoolSize;
    constexpr uint32_t kColors = config::Rule::kColors;

    // [除算・乗算の完全排除] 
    // kChunk が 64 であるため、(target_idx + 64) & ~63 と等価になり、
    // コンパイラは単一の ADD命令 と AND命令（実質1〜2クロック）のみで切り上げを完結させます。
    const uint32_t new_count =
        std::min((target_idx + kChunk) & ~(kChunk - 1u), kPoolSize);

    const uint32_t color_mask = kColors - 1u;
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
