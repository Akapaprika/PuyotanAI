#pragma once

#include <puyotan/common/types.hpp>

namespace puyotan {

struct TurnStats {
    int delta_score = 0;
    int chain_count = 0;
};

class RewardCalculator {
public:
    RewardCalculator() = default;

    // Configurable parameters
    float turn_penalty = -0.01f;
    float ojama_scale = 500.0f;
    float win_reward = 25.0f;
    float loss_reward = -25.0f;
    float draw_reward = -12.5f;

    float calculate(const TurnStats& stats, MatchStatus cur_status, int player_id) const {
        float r = turn_penalty + static_cast<float>(stats.delta_score) / ojama_scale;
        
        if (player_id == 0) {
            if (cur_status == MatchStatus::WIN_P1) r += win_reward;
            else if (cur_status == MatchStatus::WIN_P2) r += loss_reward;
            else if (cur_status == MatchStatus::DRAW) r += draw_reward;
        } else {
            if (cur_status == MatchStatus::WIN_P2) r += win_reward;
            else if (cur_status == MatchStatus::WIN_P1) r += loss_reward;
            else if (cur_status == MatchStatus::DRAW) r += draw_reward;
        }
        
        return r;
    }
};

} // namespace puyotan
