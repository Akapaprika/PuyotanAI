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
};

} // namespace puyotan::search