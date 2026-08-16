#pragma once

#include <vector>
#include <puyotan/search/eval_weights.hpp>

namespace puyotan::search {

/**
 * @struct BeamSearchSession
 * @brief Tracks multi-turn search state (such as expected score history for stagnation detection).
 */
struct BeamSearchSession {
    std::vector<float> score_history;
    int max_history_size = 10;
    int min_history_to_check = 4;
    int total_puyos_threshold = 66;    // 6 cols × 11 rows: field is near-full; stagnation is dangerous
    float growth_threshold = 0.5f;

    void update(float expected_score) {
        score_history.push_back(expected_score);
        if (static_cast<int>(score_history.size()) > max_history_size) {
            score_history.erase(score_history.begin());
        }
    }

    [[nodiscard]] bool isStagnated(int total_puyos) const noexcept {
        if (total_puyos >= total_puyos_threshold &&
            static_cast<int>(score_history.size()) >= min_history_to_check) {
            const float growth = score_history.back() - score_history[score_history.size() - min_history_to_check];
            return growth <= growth_threshold;
        }
        return false;
    }

    void reset() noexcept {
        score_history.clear();
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
