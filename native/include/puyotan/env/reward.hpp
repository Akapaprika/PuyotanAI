#pragma once

#include <puyotan/common/types.hpp>
#include <external/nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>

namespace puyotan {

class PuyotanMatch;

using json = nlohmann::json;

/**
 * @struct RewardContext
 * @brief Comprehensive snapshots of both players' states to compute complex rewards.
 */
struct RewardContext {
    // Player 1 (subject)
    int p1_delta_score = 0;
    int p1_chain_count = 0;
    int p1_puyo_count = 0;
    int p1_connectivity_score = 0;
    int p1_isolated_puyo_count = 0;
    int p1_death_col_height = 0;
    int p1_color_diversity = 0;
    int p1_buried_puyo_count = 0;
    int p1_ojama_dropped = 0;
    int p1_pending_ojama = 0;
    int p1_potential_chain = 0;   // Max chain by adding 1 puyo
    
    // Player 2 (opponent)
    int p2_delta_score = 0;
    int p2_chain_count = 0;
    int p2_puyo_count = 0;
    int p2_connectivity_score = 0;
    int p2_isolated_puyo_count = 0;
    int p2_death_col_height = 0;
    int p2_color_diversity = 0;
    int p2_buried_puyo_count = 0;
    int p2_ojama_dropped = 0;
    int p2_pending_ojama = 0;
    int p2_potential_chain = 0;

    MatchStatus status = MatchStatus::Playing;
};

struct RewardWeights {
    struct Match {
        float win = 0.0f;
        float loss = 0.0f;
        float draw = 0.0f;
    } match;

    struct Turn {
        float step_penalty = 0.0f;
    } turn;

    struct Performance {
        float score_scale = 0.0f;
        float chain_bonus_scale = 0.0f;
    } performance;

    struct Board {
        float puyo_count_penalty = 0.0f;
        float connectivity_bonus = 0.0f;
        float isolated_puyo_penalty = 0.0f;
        float death_col_height_penalty = 0.0f;
        float color_diversity_reward = 0.0f;
        float buried_puyo_penalty = 0.0f;
        float ojama_drop_penalty = 0.0f;
        float potential_chain_bonus_scale = 0.0f;
    } board;

    struct Opponent {
        float field_pressure_reward = 0.0f;
        float connectivity_penalty = 0.0f;
        float ojama_diff_scale = 0.0f;
        float initiative_bonus = 0.0f;
    } opponent;

    void from_json(const json& j) {
        if (j.contains("match")) {
            auto m = j["match"];
            if (m.contains("win")) match.win = m["win"];
            if (m.contains("loss")) match.loss = m["loss"];
            if (m.contains("draw")) match.draw = m["draw"];
        }
        if (j.contains("turn")) {
            auto t = j["turn"];
            if (t.contains("step_penalty")) turn.step_penalty = t["step_penalty"];
        }
        if (j.contains("performance")) {
            auto p = j["performance"];
            if (p.contains("score_scale")) performance.score_scale = p["score_scale"];
            if (p.contains("chain_bonus_scale")) performance.chain_bonus_scale = p["chain_bonus_scale"];
        }
        if (j.contains("board")) {
            auto b = j["board"];
            if (b.contains("puyo_count_penalty")) board.puyo_count_penalty = b["puyo_count_penalty"];
            if (b.contains("connectivity_bonus")) board.connectivity_bonus = b["connectivity_bonus"];
            if (b.contains("isolated_puyo_penalty")) board.isolated_puyo_penalty = b["isolated_puyo_penalty"];
            if (b.contains("death_col_height_penalty")) board.death_col_height_penalty = b["death_col_height_penalty"];
            if (b.contains("color_diversity_reward")) board.color_diversity_reward = b["color_diversity_reward"];
            if (b.contains("buried_puyo_penalty")) board.buried_puyo_penalty = b["buried_puyo_penalty"];
            if (b.contains("ojama_drop_penalty")) board.ojama_drop_penalty = b["ojama_drop_penalty"];
            if (b.contains("potential_chain_bonus_scale")) board.potential_chain_bonus_scale = b["potential_chain_bonus_scale"];
        }
        if (j.contains("opponent")) {
            auto o = j["opponent"];
            if (o.contains("field_pressure_reward")) opponent.field_pressure_reward = o["field_pressure_reward"];
            if (o.contains("connectivity_penalty")) opponent.connectivity_penalty = o["connectivity_penalty"];
            if (o.contains("ojama_diff_scale")) opponent.ojama_diff_scale = o["ojama_diff_scale"];
            if (o.contains("initiative_bonus")) opponent.initiative_bonus = o["initiative_bonus"];
        }
    }
};

class RewardCalculator {
public:
    RewardWeights weights;

    void load_from_json(const std::string& path_str) {
        try {
            // std::filesystem::path は Windows 上で UTF-8 文字列から適切な Wide String パスを生成する
            std::filesystem::path p = std::filesystem::u8path(path_str);
            std::ifstream f(p);
            
            if (!f.is_open()) {
                std::cerr << "[WARNING] Failed to open reward config file: " << path_str << std::endl;
                return;
            }
            json j = json::parse(f);
            weights.from_json(j);
            std::cout << "[RewardCalculator] Loaded reward config from: " << path_str << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to load/parse reward JSON: " << e.what() << " (Path: " << path_str << ")" << std::endl;
        }
    }

    void load_from_json_string(const std::string& json_str) {
        try {
            json j = json::parse(json_str);
            weights.from_json(j);
        } catch (...) {}
    }

    RewardContext extractContext(const PuyotanMatch& m, 
                                 int start_score_p1, int start_score_p2,
                                 int pre_ojama_p1, int pre_ojama_p2) const;

    float calculate(const RewardContext& ctx, int player_id) const {
        bool is_p1 = (player_id == 0);
        const int s_score  = is_p1 ? ctx.p1_delta_score : ctx.p2_delta_score;
        const int s_chain  = is_p1 ? ctx.p1_chain_count : ctx.p2_chain_count;
        const int s_puyo   = is_p1 ? ctx.p1_puyo_count  : ctx.p2_puyo_count;
        const int s_conn   = is_p1 ? ctx.p1_connectivity_score : ctx.p2_connectivity_score;
        const int s_iso    = is_p1 ? ctx.p1_isolated_puyo_count : ctx.p2_isolated_puyo_count;
        const int s_death  = is_p1 ? ctx.p1_death_col_height : ctx.p2_death_col_height;
        const int s_div    = is_p1 ? ctx.p1_color_diversity : ctx.p2_color_diversity;
        const int s_buried = is_p1 ? ctx.p1_buried_puyo_count : ctx.p2_buried_puyo_count;
        const int s_oj_drop = is_p1 ? ctx.p1_ojama_dropped : ctx.p2_ojama_dropped;
        const int s_potential = is_p1 ? ctx.p1_potential_chain : ctx.p2_potential_chain;

        const int o_score  = is_p1 ? ctx.p2_delta_score : ctx.p1_delta_score;
        const int o_puyo   = is_p1 ? ctx.p2_puyo_count  : ctx.p1_puyo_count;
        const int o_conn   = is_p1 ? ctx.p2_connectivity_score : ctx.p1_connectivity_score;
        const int o_potential = is_p1 ? ctx.p2_potential_chain : ctx.p1_potential_chain;

        float r = 0.0f;

        r += weights.turn.step_penalty;
        if (ctx.status != MatchStatus::Playing) {
            bool win = (is_p1 && ctx.status == MatchStatus::WinP1) || (!is_p1 && ctx.status == MatchStatus::WinP2);
            bool loss = (is_p1 && ctx.status == MatchStatus::WinP2) || (!is_p1 && ctx.status == MatchStatus::WinP1);
            if (win) r += weights.match.win;
            else if (loss) r += weights.match.loss;
            else r += weights.match.draw;
        }

        r += static_cast<float>(s_score) * weights.performance.score_scale;
        if (s_chain > 0) {
            r += static_cast<float>(s_chain * s_chain) * weights.performance.chain_bonus_scale;
        }

        r += static_cast<float>(s_puyo)  * weights.board.puyo_count_penalty;
        r += static_cast<float>(s_conn)  * weights.board.connectivity_bonus;
        r += static_cast<float>(s_iso)   * weights.board.isolated_puyo_penalty;
        r += static_cast<float>(s_death) * weights.board.death_col_height_penalty;
        r += static_cast<float>(s_div)   * weights.board.color_diversity_reward;
        r += static_cast<float>(s_buried) * weights.board.buried_puyo_penalty;
        r += static_cast<float>(s_oj_drop) * weights.board.ojama_drop_penalty;
        
        if (s_potential > 0) {
            r += static_cast<float>(s_potential * s_potential) * weights.board.potential_chain_bonus_scale;
        }

        r += static_cast<float>(o_puyo) * weights.opponent.field_pressure_reward;
        r += static_cast<float>(o_conn) * weights.opponent.connectivity_penalty;
        r += static_cast<float>(s_score - o_score) * weights.opponent.ojama_diff_scale; 

        if (s_potential > 0 && o_potential == 0) {
            r += weights.opponent.initiative_bonus;
        }

        return r;
    }
};

} // namespace puyotan
