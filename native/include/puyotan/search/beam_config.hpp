#pragma once

#include <array>
#include <puyotan/search/eval_weights.hpp>

namespace puyotan::search {

/**
 * @struct BeamSearchSession
 * @brief Tracks multi-turn search state (such as expected score history for stagnation detection).
 *
 * Uses a fixed-size ring buffer (std::array + head index) for O(1) update instead of
 * the previous O(n) std::vector::erase(begin()) pattern.
 */
struct BeamSearchSession {
    int min_history_to_check = 4;
    int total_puyos_threshold = 66;    // 6 cols × 11 rows: field is near-full
    float growth_threshold    = 0.5f;

  private:
    static constexpr int kMaxHistory = 10;
    std::array<float, kMaxHistory> buf_{};
    int head_ = 0;  ///< Index of the slot that will be written next
    int size_ = 0;  ///< Number of valid entries currently stored

  public:
    void update(float expected_score) noexcept {
        buf_[head_] = expected_score;
        head_ = (head_ + 1) % kMaxHistory;
        if (size_ < kMaxHistory) ++size_;
    }

    [[nodiscard]] bool isStagnated(int total_puyos) const noexcept {
        if (total_puyos >= total_puyos_threshold && size_ >= min_history_to_check) {
            // newest entry  : buf_[(head_ - 1 + kMaxHistory) % kMaxHistory]
            // entry min_history_to_check steps ago: buf_[(head_ - min_history_to_check + kMaxHistory) % kMaxHistory]
            const float newest = buf_[(head_ - 1 + kMaxHistory) % kMaxHistory];
            const float oldest = buf_[(head_ - min_history_to_check + kMaxHistory) % kMaxHistory];
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
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    int full_beam_depth = 2;
    float min_beam_width_ratio = 1.0f;
    SoloBeamEvalWeights eval_weights;
};

/**
 * @struct VsBeamConfig
 * @brief Parameters controlling VS beam search behaviour.
 */
struct VsBeamConfig {
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    int full_beam_depth = 2;
    float min_beam_width_ratio = 1.0f;
    bool enable_attack_search = true;
    VsBeamEvalWeights eval_weights;
    VsEvalContext     context;
};

} // namespace puyotan::search
