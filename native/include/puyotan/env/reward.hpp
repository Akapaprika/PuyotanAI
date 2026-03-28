#pragma once

#include <puyotan/common/types.hpp>

namespace puyotan {

/**
 * @struct TurnStats
 * @brief Captures essential metrics from a single frame to compute RL rewards.
 */
struct TurnStats {
    int delta_score = 0; ///< Points gained in this specific step
    int chain_count = 0; ///< Current chain length (if any)
};

/**
 * @class RewardCalculator
 * @brief Logic for calculating float-valued rewards for RL agents.
 */
class RewardCalculator {
public:
    RewardCalculator() = default;

    // Configurable parameters
    float turn_penalty = -0.01f; ///< Penalty per frame to encourage faster wins
    float ojama_scale = 500.0f;  ///< Scale factor for score-to-reward conversion
    float win_reward = 25.0f;    ///< Flat bonus for winning
    float loss_reward = -25.0f;  ///< Flat penalty for losing
    float draw_reward = -12.5f;  ///< Flat penalty for a draw

    /**
     * @brief Computes the reward for a specific player based on turn stats and game status.
     * @param stats Metrics from the current step.
     * @param cur_status Final result of the match.
     * @param player_id The player index (0 or 1).
     * @return Calculated float reward.
     */
    float calculate(const TurnStats& stats, MatchStatus cur_status, int player_id) const {
        float r = turn_penalty + static_cast<float>(stats.delta_score) / ojama_scale;
        
        if (player_id == 0) {
            if (cur_status == MatchStatus::WinP1) r += win_reward;
            else if (cur_status == MatchStatus::WinP2) r += loss_reward;
            else if (cur_status == MatchStatus::Draw) r += draw_reward;
        } else {
            if (cur_status == MatchStatus::WinP2) r += win_reward;
            else if (cur_status == MatchStatus::WinP1) r += loss_reward;
            else if (cur_status == MatchStatus::Draw) r += draw_reward;
        }
        
        return r;
    }
};

} // namespace puyotan
