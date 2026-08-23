#pragma once

#include <array>
#include <puyotan/search/eval_weights.hpp>

namespace puyotan::search {

/**
 * @struct BeamSearchSession
 * @brief Tracks multi-turn search state (such as expected score history for stagnation detection).
 *
 * Uses a power-of-two fixed-size ring buffer (size 16) with bitmask indexing for zero-cost O(1) updates.
 */
struct BeamSearchSession {
    int min_history_to_check  = 4;
    int total_puyos_threshold = 66;    // 6 cols × 11 rows: field is near-full
    int32_t growth_threshold  = 500;   // Less than 500 pts growth over min_history_to_check is considered stagnation

  private:
    static constexpr int kMaxHistory = 16;           ///< 2のべき乗（16）にしてビットマスク化
    static constexpr int kMask       = kMaxHistory - 1;

    std::array<int32_t, kMaxHistory> buf_{};
    int head_ = 0;  ///< Index of the slot that will be written next
    int size_ = 0;  ///< Number of valid entries currently stored

  public:
    void update(int32_t expected_score) noexcept {
        buf_[head_] = expected_score;
        head_ = (head_ + 1) & kMask;
        if (size_ < kMaxHistory) ++size_;
    }

    [[nodiscard]] bool isStagnated(int total_puyos) const noexcept {
        if (total_puyos >= total_puyos_threshold && size_ >= min_history_to_check) {
            // ★ 剰余演算や + kMaxHistory を排除し、1命令の & kMask で完結
            const int32_t newest = buf_[(head_ - 1) & kMask];
            const int32_t oldest = buf_[(head_ - min_history_to_check) & kMask];
            return (newest - oldest) <= growth_threshold;
        }
        return false;
    }

    void reset() noexcept {
        head_ = 0;
        size_ = 0;
    }
};

/**
 * @struct SoloBeamConfig
 * @brief Parameters controlling solo beam search behaviour.
 */
struct SoloBeamConfig {
    int   beam_width           = 500;
    int   look_ahead           = 3;
    int   dbs_max_similar      = 0;
    int   full_beam_depth      = 2;
    float min_beam_width_ratio = 1.0f;
    SoloBeamEvalWeights eval_weights;
    std::array<int, 64> target_beam_widths{};

    SoloBeamConfig() noexcept { recompute_beam_widths(); }

    void recompute_beam_widths() noexcept {
        look_ahead = std::clamp(look_ahead, 1, 64);
        dbs_max_similar = std::clamp(dbs_max_similar, 0, 65535);
        const int max_lookahead = look_ahead;
        if (min_beam_width_ratio < 1.0f && look_ahead > 1) {
            const float max_decay_steps = static_cast<float>(look_ahead - 1 - full_beam_depth);
            const float inv_decay = (max_decay_steps > 0.0f) ? (1.0f / max_decay_steps) : 0.0f;
            for (int d = 0; d < max_lookahead; ++d) {
                if (d <= full_beam_depth) {
                    target_beam_widths[d] = beam_width;
                } else {
                    const float progress = static_cast<float>(d - full_beam_depth) * inv_decay;
                    const float ratio = 1.0f - (1.0f - min_beam_width_ratio) * progress;
                    target_beam_widths[d] = std::max(1, static_cast<int>(beam_width * ratio));
                }
            }
        } else {
            target_beam_widths.fill(beam_width);
        }
    }
};

/**
 * @struct VsBeamConfig
 * @brief Parameters controlling VS beam search behaviour.
 */
struct VsBeamConfig {
    int   beam_width           = 500;
    int   look_ahead           = 3;
    int   dbs_max_similar      = 0;
    int   full_beam_depth      = 2;
    float min_beam_width_ratio = 1.0f;
    bool  enable_attack_search = true;
    VsBeamEvalWeights eval_weights;
    VsEvalContext     context;
    std::array<int, 64> target_beam_widths{};

    VsBeamConfig() noexcept { recompute_beam_widths(); }

    void recompute_beam_widths() noexcept {
        look_ahead = std::clamp(look_ahead, 1, 64);
        dbs_max_similar = std::clamp(dbs_max_similar, 0, 65535);
        const int max_lookahead = look_ahead;
        if (min_beam_width_ratio < 1.0f && look_ahead > 1) {
            const float max_decay_steps = static_cast<float>(look_ahead - 1 - full_beam_depth);
            const float inv_decay = (max_decay_steps > 0.0f) ? (1.0f / max_decay_steps) : 0.0f;
            for (int d = 0; d < max_lookahead; ++d) {
                if (d <= full_beam_depth) {
                    target_beam_widths[d] = beam_width;
                } else {
                    const float progress = static_cast<float>(d - full_beam_depth) * inv_decay;
                    const float ratio = 1.0f - (1.0f - min_beam_width_ratio) * progress;
                    target_beam_widths[d] = std::max(1, static_cast<int>(beam_width * ratio));
                }
            }
        } else {
            target_beam_widths.fill(beam_width);
        }
    }
};

} // namespace puyotan::search