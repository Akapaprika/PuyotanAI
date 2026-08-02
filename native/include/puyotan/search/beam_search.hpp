#pragma once

#include <vector>
#include <utility>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @struct BeamSearchSession
 * @brief Tracks multi-turn search state (such as expected score history for stagnation detection).
 */
struct BeamSearchSession {
    std::vector<float> score_history;
    int max_history_size = 10;
    int min_history_to_check = 4;
    int total_puyos_threshold = 66;
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
    SoloBeamEvalWeights eval_weights;
};

/**
 * @struct VsBeamConfig
 * @brief Parameters controlling VS beam search behaviour.
 *
 * Populate context from the live match state before every call to
 * vsBeamSearch().
 */
struct VsBeamConfig {
    int beam_width = 500;
    int look_ahead = 3;
    int dbs_max_similar = 0;
    bool enable_attack_search = true;
    VsBeamEvalWeights eval_weights;
    VsEvalContext     context;                          // set before each call
};

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for Solo mode.
 */
std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const TsumoSequence& tsumo_seq,
                                     const SoloBeamConfig& cfg,
                                     BeamSearchSession*   session = nullptr) noexcept;

/**
 * @brief Runs a beam search from the given player state and returns the best RL action index and its expected score for VS mode.
 */
std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const TsumoSequence& tsumo_seq,
                                   const VsBeamConfig&  cfg,
                                   BeamSearchSession*   session = nullptr) noexcept;

/**
 * @brief Runs a VS beam search and returns the top N unique candidate actions with their scores.
 */
std::vector<std::pair<int, float>> vsBeamSearchTopN(const PuyotanPlayer& player,
                                                    const TsumoSequence& tsumo_seq,
                                                    const VsBeamConfig&  cfg,
                                                    int                  top_n = 5) noexcept;

} // namespace puyotan::search
