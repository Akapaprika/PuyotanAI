#include <puyotan/search/abs_search.hpp>
#include <puyotan/search/negamax_search.hpp>
#include <algorithm>
#include <unordered_map>
#include <omp.h>

namespace puyotan::search {

struct AbsNode {
    PuyotanMatch match;
    float score = 0.0f;
    int first_action = -1;
    AbsCategory category = AbsCategory::Build;
};

static AbsCategory classifyMove(const PuyotanMatch& prev_match,
                                 const PuyotanMatch& next_match,
                                 int active_player_id) noexcept {
    const auto& prev_p = prev_match.getPlayer(active_player_id);
    const auto& next_p = next_match.getPlayer(active_player_id);
    const auto& opp_p  = next_match.getPlayer(1 - active_player_id);

    const int score_diff = next_p.score - prev_p.score;
    const int total_incoming = prev_p.active_ojama + prev_p.non_active_ojama;

    // 1. EVADE: Incoming ojama existed and was offset/cleared by chain
    if (total_incoming > 0 && score_diff > 0) {
        return AbsCategory::Evade;
    }

    // 2. CRUSH: Chain occurred, enemy not in chain animation, >= 6 ojama sent
    if (score_diff > 0 && opp_p.current_action.action.type != ActionType::Chain) {
        const int sent_ojama = (next_p.used_score - prev_p.used_score) / config::Score::kTargetScore;
        if (sent_ojama >= 6) {
            return AbsCategory::Crush;
        }
    }

    // 3. STRIKE: Any other chain trigger
    if (score_diff > 0) {
        return AbsCategory::Strike;
    }

    // 4. BUILD: No chain trigger
    return AbsCategory::Build;
}

static void pruneByCategory(std::vector<AbsNode>& candidates,
                            const CategoryBudgets& budgets,
                            bool is_my_turn,
                            std::vector<AbsNode>& out_beam) {
    std::vector<AbsNode> builds, crushes, strikes, evades;
    builds.reserve(candidates.size());
    crushes.reserve(candidates.size() / 4);
    strikes.reserve(candidates.size() / 4);
    evades.reserve(candidates.size() / 4);

    for (auto& node : candidates) {
        switch (node.category) {
            case AbsCategory::Build:  builds.push_back(std::move(node)); break;
            case AbsCategory::Crush:  crushes.push_back(std::move(node)); break;
            case AbsCategory::Strike: strikes.push_back(std::move(node)); break;
            case AbsCategory::Evade:  evades.push_back(std::move(node)); break;
        }
    }

    auto comp = [is_my_turn](const AbsNode& a, const AbsNode& b) {
        return is_my_turn ? (a.score > b.score) : (a.score < b.score);
    };

    std::sort(builds.begin(),  builds.end(),  comp);
    std::sort(crushes.begin(), crushes.end(), comp);
    std::sort(strikes.begin(), strikes.end(), comp);
    std::sort(evades.begin(), evades.end(), comp);

    out_beam.clear();

    auto append_top = [&](std::vector<AbsNode>& src, int budget) {
        const int count = std::min(static_cast<int>(src.size()), budget);
        for (int i = 0; i < count; ++i) {
            out_beam.push_back(std::move(src[i]));
        }
    };

    append_top(builds,  budgets.build);
    append_top(crushes, budgets.crush);
    append_top(strikes, budgets.strike);
    append_top(evades,  budgets.evade);
}

AbsResult absSearch(const PuyotanMatch& match, int my_id, const AbsConfig& cfg) noexcept {
    AbsResult res;
    if (match.getStatus() != MatchStatus::Playing) {
        res.best_eval = evaluateMatchState(match, my_id, cfg.eval_weights);
        return res;
    }

    std::vector<AbsNode> current_beam;
    current_beam.push_back(AbsNode{match, 0.0f, -1, AbsCategory::Build});

    const int max_threads = omp_get_max_threads();

    for (int step = 0; step < cfg.depth; ++step) {
        if (current_beam.empty()) break;

        const int num_parents = static_cast<int>(current_beam.size());
        std::vector<std::vector<AbsNode>> thread_local_candidates(max_threads);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local_cands = thread_local_candidates[tid];

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < num_parents; ++i) {
                const auto& parent = current_beam[i];

                if (parent.match.getStatus() != MatchStatus::Playing) {
                    local_cands.push_back(parent);
                    continue;
                }

                PuyotanMatch m = parent.match;
                int mask = m.getDecisionMask();
                if (mask == 0) {
                    mask = m.stepUntilDecision();
                    if (m.getStatus() != MatchStatus::Playing || mask == 0) {
                        AbsNode n = parent;
                        n.match = m;
                        n.score = evaluateMatchState(m, my_id, cfg.eval_weights);
                        local_cands.push_back(n);
                        continue;
                    }
                }

                // Determine active player
                const int active_player = (mask & 1) ? 0 : 1;

                // Expand 22 RL actions for active player
                for (int act_idx = 0; act_idx < kNumRLActions; ++act_idx) {
                    PuyotanMatch child_match = m;
                    child_match.setAction(active_player, getRLAction(act_idx));
                    child_match.stepUntilDecision();

                    const AbsCategory cat = classifyMove(m, child_match, active_player);
                    float score = evaluateMatchState(child_match, my_id, cfg.eval_weights);

                    // --- CRITICAL ATTACK / FIRE SCORE ADJUSTMENT ---
                    // When a player fires a chain (score_diff > 0), potential drops because puyos disappear.
                    // We must scale score_diff by fire_bias to properly value the damage sent to the opponent.
                    const auto& parent_act_p = m.getPlayer(active_player);
                    const auto& child_act_p = child_match.getPlayer(active_player);
                    const int score_diff = child_act_p.score - parent_act_p.score;

                    if (score_diff > 0) {
                        const float attack_bonus = static_cast<float>(score_diff) * cfg.eval_weights.fire_bias * 2.5f;
                        score += (active_player == my_id) ? attack_bonus : -attack_bonus;
                    }

                    const int first_act = (step == 0) ? act_idx : parent.first_action;
                    local_cands.push_back(AbsNode{child_match, score, first_act, cat});
                }
            }
        }

        std::vector<AbsNode> next_candidates;
        size_t total_cands = 0;
        for (const auto& vec : thread_local_candidates) {
            total_cands += vec.size();
        }
        next_candidates.reserve(total_cands);

        for (auto& vec : thread_local_candidates) {
            for (auto& n : vec) {
                next_candidates.push_back(std::move(n));
            }
        }

        if (next_candidates.empty()) break;

        // Determine step turn from first non-terminal candidate
        bool step_is_my_turn = true;
        for (const auto& n : next_candidates) {
            int m_mask = n.match.getDecisionMask();
            if (m_mask != 0) {
                step_is_my_turn = (((m_mask & 1) ? 0 : 1) == my_id);
                break;
            }
        }

        const auto& budgets = step_is_my_turn ? cfg.my_budgets : cfg.opp_budgets;
        pruneByCategory(next_candidates, budgets, step_is_my_turn, current_beam);
    }

    if (current_beam.empty()) {
        res.best_eval = evaluateMatchState(match, my_id, cfg.eval_weights);
        return res;
    }

    // Find best node in final beam
    float best_eval = -1e9f;
    int best_act = current_beam[0].first_action;

    std::unordered_map<int, float> best_eval_per_action;

    for (const auto& node : current_beam) {
        if (node.first_action < 0) continue;
        auto it = best_eval_per_action.find(node.first_action);
        if (it == best_eval_per_action.end() || node.score > it->second) {
            best_eval_per_action[node.first_action] = node.score;
        }
        if (node.score > best_eval) {
            best_eval = node.score;
            best_act = node.first_action;
        }
    }

    for (const auto& [act, eval_val] : best_eval_per_action) {
        res.candidate_evals.emplace_back(act, eval_val);
    }

    res.best_action = (best_act >= 0) ? best_act : 0;
    res.best_eval = (best_eval > -1e8f) ? best_eval : evaluateMatchState(match, my_id, cfg.eval_weights);
    return res;
}

} // namespace puyotan::search
