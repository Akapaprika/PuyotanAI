#pragma once

#include <puyotan/common/types.hpp>
#include <external/nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>
#include <cmath>

namespace puyotan {

class PuyotanMatch;

using json = nlohmann::json;

/**
 * @struct RewardContext
 * @brief Snapshot of both players' game events for a single step.
 *
 * Populated by RewardCalculator::extractContext().
 * The reward layer reads this; it does NOT compute rewards itself.
 *
 * Naming convention:
 *   p1_* = subject player (the agent being trained)
 *   p2_* = opponent
 */
struct RewardContext {
    // -----------------------------------------------------------------------
    // Player 1 (subject)
    // -----------------------------------------------------------------------
    // Performance
    int p1_delta_score      = 0;  ///< Score gained this step (includes soft-drop)
    int p1_chain_count      = 0;  ///< Chains completed this step (last_chain_count)
    int p1_total_erased     = 0;  ///< Total puyos erased in the chain sequence this step
    int p1_ojama_sent       = 0;  ///< Ojama puyos sent to opponent this step (delta_score / kTargetScore)
    bool p1_all_clear       = false; ///< True if the board was fully cleared this step

    // Board state
    int p1_puyo_count            = 0;  ///< Total puyos on field
    int p1_connectivity_score    = 0;  ///< Puyos with ≥2 same-color neighbors
    int p1_isolated_puyo_count   = 0;  ///< Puyos with 0 same-color neighbors
    int p1_near_group_count      = 0;  ///< Same-color groups of size 3 (one away from firing)
    int p1_death_col_height      = 0;  ///< Height of death column (col kDeathCol)
    float p1_height_variance     = 0.0f; ///< Variance of column heights (uniformity)
    int p1_color_diversity       = 0;  ///< Number of colors with connected groups
    int p1_buried_puyo_count     = 0;  ///< Colored puyos buried under ojama
    int p1_ojama_dropped         = 0;  ///< Ojama that fell ON p1 this step
    int p1_pending_ojama         = 0;  ///< Incoming ojama not yet fallen (active + non_active)
    int p1_potential_chain       = 0;  ///< Max chain achievable by adding 1 puyo

    // -----------------------------------------------------------------------
    // Player 2 (opponent) — symmetric
    // -----------------------------------------------------------------------
    int p2_delta_score           = 0;
    int p2_chain_count           = 0;
    int p2_total_erased          = 0;
    int p2_ojama_sent            = 0;
    bool p2_all_clear            = false;

    int p2_puyo_count            = 0;
    int p2_connectivity_score    = 0;
    int p2_isolated_puyo_count   = 0;
    int p2_near_group_count      = 0;
    int p2_death_col_height      = 0;
    float p2_height_variance     = 0.0f;
    int p2_color_diversity       = 0;
    int p2_buried_puyo_count     = 0;
    int p2_ojama_dropped         = 0;
    int p2_pending_ojama         = 0;
    int p2_potential_chain       = 0;

    MatchStatus status = MatchStatus::Playing;
};

/**
 * @struct RewardWeights
 * @brief Tunable parameters for reward calculation. Loaded from JSON.
 *
 * Design intent:
 *   - Solo mode:  chain_power=3, match weights=0, opponent weights=0
 *   - Match mode: chain_power=2, match weights active, opponent weights active
 */
struct RewardWeights {
    struct Match {
        float win  = 0.0f;
        float loss = 0.0f;
        float draw = 0.0f;
    } match;

    struct Turn {
        float step_penalty = 0.0f;
    } turn;

    struct Performance {
        float score_scale              = 0.0f; ///< Scale for delta_score
        float chain_scale              = 0.0f; ///< Base scale for chain reward
        float chain_power              = 2.0f; ///< Exponent: reward = chain^power * chain_scale
        float min_chain_threshold      = 0.0f; ///< Chains below this apply premature_chain_penalty
        float premature_chain_penalty  = 0.0f; ///< Penalty per chain below threshold (usually negative)
        float all_clear_bonus          = 0.0f; ///< Flat bonus for achieving All Clear
        float erasure_count_scale      = 0.0f; ///< Reward per puyo erased in chain sequence
        float ojama_sent_scale         = 0.0f; ///< Reward per ojama puyo sent to opponent
    } performance;

    struct Board {
        float puyo_count_penalty         = 0.0f;
        float connectivity_bonus         = 0.0f;
        float isolated_puyo_penalty      = 0.0f;
        float near_group_bonus           = 0.0f; ///< Reward per 3-puyo same-color group
        float height_variance_penalty    = 0.0f; ///< Penalty for uneven column heights
        float death_col_height_penalty   = 0.0f;
        float color_diversity_reward     = 0.0f;
        float buried_puyo_penalty        = 0.0f;
        float ojama_drop_penalty         = 0.0f;
        float pending_ojama_penalty      = 0.0f; ///< Penalty for incoming ojama not yet fallen
        float potential_chain_bonus_scale= 0.0f;
    } board;

    struct Opponent {
        float field_pressure_reward  = 0.0f;
        float connectivity_penalty   = 0.0f;
        float ojama_diff_scale       = 0.0f; ///< Legacy alias — prefer ojama_sent_scale in Performance
        float initiative_bonus       = 0.0f;
    } opponent;

    void from_json(const json& j) {
        auto get = [](const json& obj, const char* key, float& dst) {
            if (obj.contains(key)) dst = obj[key].get<float>();
        };
        if (j.contains("match")) {
            const auto& m = j["match"];
            get(m, "win",  match.win);
            get(m, "loss", match.loss);
            get(m, "draw", match.draw);
        }
        if (j.contains("turn")) {
            get(j["turn"], "step_penalty", turn.step_penalty);
        }
        if (j.contains("performance")) {
            const auto& p = j["performance"];
            get(p, "score_scale",              performance.score_scale);
            get(p, "chain_scale",              performance.chain_scale);
            get(p, "chain_bonus_scale",        performance.chain_scale); // legacy alias
            get(p, "chain_power",              performance.chain_power);
            get(p, "min_chain_threshold",      performance.min_chain_threshold);
            get(p, "premature_chain_penalty",  performance.premature_chain_penalty);
            get(p, "all_clear_bonus",          performance.all_clear_bonus);
            get(p, "erasure_count_scale",      performance.erasure_count_scale);
            get(p, "ojama_sent_scale",         performance.ojama_sent_scale);
        }
        if (j.contains("board")) {
            const auto& b = j["board"];
            get(b, "puyo_count_penalty",          board.puyo_count_penalty);
            get(b, "connectivity_bonus",          board.connectivity_bonus);
            get(b, "isolated_puyo_penalty",       board.isolated_puyo_penalty);
            get(b, "near_group_bonus",            board.near_group_bonus);
            get(b, "height_variance_penalty",     board.height_variance_penalty);
            get(b, "death_col_height_penalty",    board.death_col_height_penalty);
            get(b, "color_diversity_reward",      board.color_diversity_reward);
            get(b, "buried_puyo_penalty",         board.buried_puyo_penalty);
            get(b, "ojama_drop_penalty",          board.ojama_drop_penalty);
            get(b, "pending_ojama_penalty",       board.pending_ojama_penalty);
            get(b, "potential_chain_bonus_scale", board.potential_chain_bonus_scale);
        }
        if (j.contains("opponent")) {
            const auto& o = j["opponent"];
            get(o, "field_pressure_reward", opponent.field_pressure_reward);
            get(o, "connectivity_penalty",  opponent.connectivity_penalty);
            get(o, "ojama_diff_scale",      opponent.ojama_diff_scale);
            get(o, "initiative_bonus",      opponent.initiative_bonus);
        }
    }
};

class RewardCalculator {
public:
    RewardWeights weights;

    void load_from_json(const std::string& path_str) {
        try {
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
            std::cerr << "[ERROR] Failed to load/parse reward JSON: " << e.what()
                      << " (Path: " << path_str << ")" << std::endl;
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
                                 int pre_ojama_p1,   int pre_ojama_p2) const;

    float calculate(const RewardContext& ctx, int player_id) const {
        const bool is_p1 = (player_id == 0);

        // ---- Subject ----
        const int   s_score     = is_p1 ? ctx.p1_delta_score   : ctx.p2_delta_score;
        const int   s_chain     = is_p1 ? ctx.p1_chain_count   : ctx.p2_chain_count;
        const int   s_erased    = is_p1 ? ctx.p1_total_erased  : ctx.p2_total_erased;
        const int   s_oj_sent   = is_p1 ? ctx.p1_ojama_sent    : ctx.p2_ojama_sent;
        const bool  s_all_clear = is_p1 ? ctx.p1_all_clear     : ctx.p2_all_clear;
        const int   s_puyo      = is_p1 ? ctx.p1_puyo_count    : ctx.p2_puyo_count;
        const int   s_conn      = is_p1 ? ctx.p1_connectivity_score  : ctx.p2_connectivity_score;
        const int   s_iso       = is_p1 ? ctx.p1_isolated_puyo_count : ctx.p2_isolated_puyo_count;
        const int   s_near      = is_p1 ? ctx.p1_near_group_count    : ctx.p2_near_group_count;
        const float s_hvar      = is_p1 ? ctx.p1_height_variance     : ctx.p2_height_variance;
        const int   s_death     = is_p1 ? ctx.p1_death_col_height    : ctx.p2_death_col_height;
        const int   s_div       = is_p1 ? ctx.p1_color_diversity     : ctx.p2_color_diversity;
        const int   s_buried    = is_p1 ? ctx.p1_buried_puyo_count   : ctx.p2_buried_puyo_count;
        const int   s_oj_drop   = is_p1 ? ctx.p1_ojama_dropped       : ctx.p2_ojama_dropped;
        const int   s_pending   = is_p1 ? ctx.p1_pending_ojama       : ctx.p2_pending_ojama;
        const int   s_potential = is_p1 ? ctx.p1_potential_chain     : ctx.p2_potential_chain;

        // ---- Opponent ----
        const int o_puyo      = is_p1 ? ctx.p2_puyo_count          : ctx.p1_puyo_count;
        const int o_conn      = is_p1 ? ctx.p2_connectivity_score   : ctx.p1_connectivity_score;
        const int o_potential = is_p1 ? ctx.p2_potential_chain      : ctx.p1_potential_chain;

        float r = 0.0f;

        // Turn penalty
        r += weights.turn.step_penalty;

        // Match outcome
        if (ctx.status != MatchStatus::Playing) {
            const bool win  = (is_p1 && ctx.status == MatchStatus::WinP1)
                           || (!is_p1 && ctx.status == MatchStatus::WinP2);
            const bool loss = (is_p1 && ctx.status == MatchStatus::WinP2)
                           || (!is_p1 && ctx.status == MatchStatus::WinP1);
            if (win)       r += weights.match.win;
            else if (loss) r += weights.match.loss;
            else           r += weights.match.draw;
        }

        // Score
        r += static_cast<float>(s_score) * weights.performance.score_scale;

        // Chain reward (with power curve and threshold)
        if (s_chain > 0) {
            if (static_cast<float>(s_chain) >= weights.performance.min_chain_threshold) {
                r += std::pow(static_cast<float>(s_chain), weights.performance.chain_power)
                     * weights.performance.chain_scale;
            } else {
                r += static_cast<float>(s_chain) * weights.performance.premature_chain_penalty;
            }
        }

        // All Clear bonus
        if (s_all_clear) r += weights.performance.all_clear_bonus;

        // Erasure count bonus
        r += static_cast<float>(s_erased) * weights.performance.erasure_count_scale;

        // Ojama sent
        r += static_cast<float>(s_oj_sent) * weights.performance.ojama_sent_scale;

        // Board state
        r += static_cast<float>(s_puyo)    * weights.board.puyo_count_penalty;
        r += static_cast<float>(s_conn)    * weights.board.connectivity_bonus;
        r += static_cast<float>(s_iso)     * weights.board.isolated_puyo_penalty;
        r += static_cast<float>(s_near)    * weights.board.near_group_bonus;
        r -= s_hvar                         * weights.board.height_variance_penalty;
        r += static_cast<float>(s_death)   * weights.board.death_col_height_penalty;
        r += static_cast<float>(s_div)     * weights.board.color_diversity_reward;
        r += static_cast<float>(s_buried)  * weights.board.buried_puyo_penalty;
        r += static_cast<float>(s_oj_drop) * weights.board.ojama_drop_penalty;
        r += static_cast<float>(s_pending) * weights.board.pending_ojama_penalty;

        if (s_potential > 0) {
            r += static_cast<float>(s_potential * s_potential)
                 * weights.board.potential_chain_bonus_scale;
        }

        // Opponent
        r += static_cast<float>(o_puyo) * weights.opponent.field_pressure_reward;
        r += static_cast<float>(o_conn) * weights.opponent.connectivity_penalty;
        // Legacy ojama_diff_scale (uses score delta as proxy — kept for backwards compat)
        r += static_cast<float>(s_score - (is_p1 ? ctx.p2_delta_score : ctx.p1_delta_score))
             * weights.opponent.ojama_diff_scale;

        if (s_potential > 0 && o_potential == 0) {
            r += weights.opponent.initiative_bonus;
        }

        return r;
    }
};

} // namespace puyotan
