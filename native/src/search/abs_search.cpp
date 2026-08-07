#include <puyotan/search/abs_search.hpp>
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

// ---------------------------------------------------------------------------
// Full precision evaluation: SoloBeamEvaluator (expensive, ~24 chain sims)
// Called only on surviving nodes after pruning, NOT during expansion.
// ---------------------------------------------------------------------------
static float evaluateAbsState(const PuyotanMatch& match, int my_id,
                               const SoloBeamEvalWeights& weights) noexcept {
    const MatchStatus st = match.getStatus();
    if (st == MatchStatus::WinP1) return my_id == 0 ? 1000000.0f : -1000000.0f;
    if (st == MatchStatus::WinP2) return my_id == 1 ? 1000000.0f : -1000000.0f;
    if (st == MatchStatus::Draw)  return 0.0f;

    const auto& my_p  = match.getPlayer(my_id);
    const auto& opp_p = match.getPlayer(1 - my_id);

    const float my_val  = static_cast<float>(my_p.score)
                        + SoloBeamEvaluator::evaluate(my_p.field, weights);
    const float opp_val = static_cast<float>(opp_p.score)
                        + SoloBeamEvaluator::evaluate(opp_p.field, weights);

    return my_val - opp_val;
}

// ---------------------------------------------------------------------------
// Cheap proxy score (O(1) per node, used during expansion for pre-pruning)
// Rationale: parent.score is the full eval from the previous step.
//            We add the incremental game score gained in this step.
//            For BUILD nodes (no chain fired), proxy = parent.score.
//            For CRUSH/STRIKE nodes, proxy = parent.score + big positive.
// This gives meaningful ranking WITHOUT calling computeMaxPotentialScore.
// ---------------------------------------------------------------------------
static float proxyScore(const AbsNode& parent, const PuyotanMatch& child,
                        int my_id) noexcept {
    const auto& child_my  = child.getPlayer(my_id);
    const auto& child_opp = child.getPlayer(1 - my_id);
    const auto& par_my    = parent.match.getPlayer(my_id);
    const auto& par_opp   = parent.match.getPlayer(1 - my_id);

    const float incremental =
        static_cast<float>(child_my.score  - child_opp.score) -
        static_cast<float>(par_my.score    - par_opp.score);

    return parent.score + incremental;
}

static AbsCategory classifyMove(const PuyotanMatch& prev_match,
                                const PuyotanMatch& next_match,
                                int active_player_id) noexcept {
    const auto& prev_p = prev_match.getPlayer(active_player_id);
    const auto& next_p = next_match.getPlayer(active_player_id);
    const auto& opp_p  = next_match.getPlayer(1 - active_player_id);

    const int score_diff     = next_p.score - prev_p.score;
    const int total_incoming = prev_p.active_ojama + prev_p.non_active_ojama;

    if (total_incoming > 0 && score_diff > 0) return AbsCategory::Evade;
    if (score_diff > 0 && opp_p.current_action.action.type != ActionType::Chain) {
        const int sent_ojama =
            (next_p.used_score - prev_p.used_score) / config::Score::kTargetScore;
        if (sent_ojama >= 6) return AbsCategory::Crush;
    }
    if (score_diff > 0) return AbsCategory::Strike;
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
            case AbsCategory::Build:  builds.push_back(std::move(node));  break;
            case AbsCategory::Crush:  crushes.push_back(std::move(node)); break;
            case AbsCategory::Strike: strikes.push_back(std::move(node)); break;
            case AbsCategory::Evade:  evades.push_back(std::move(node));  break;
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
        const int n = std::min(static_cast<int>(src.size()), budget);
        for (int i = 0; i < n; ++i) out_beam.push_back(std::move(src[i]));
    };
    append_top(builds,  budgets.build);
    append_top(crushes, budgets.crush);
    append_top(strikes, budgets.strike);
    append_top(evades,  budgets.evade);
}

AbsResult absSearch(const PuyotanMatch& match, int my_id,
                    const AbsConfig& cfg) noexcept {
    AbsResult res;
    if (match.getStatus() != MatchStatus::Playing) {
        res.best_eval = evaluateAbsState(match, my_id, cfg.eval_weights);
        return res;
    }

    // --- Initial beam: root node, score=0 (no prior full eval) ---
    std::vector<AbsNode> current_beam;
    current_beam.push_back(AbsNode{match, 0.0f, -1, AbsCategory::Build});

    const int max_threads = omp_get_max_threads();

    for (int step = 0; step < cfg.depth; ++step) {
        if (current_beam.empty()) break;

        const int num_parents = static_cast<int>(current_beam.size());
        std::vector<std::vector<AbsNode>> tl_candidates(max_threads);

        // ---------------------------------------------------------------
        // Phase 1: Expand with CHEAP proxy score (O(1) per node)
        //          computeMaxPotentialScore is NOT called here.
        // ---------------------------------------------------------------
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local = tl_candidates[tid];

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < num_parents; ++i) {
                const auto& parent = current_beam[i];

                if (parent.match.getStatus() != MatchStatus::Playing) {
                    local.push_back(parent);
                    continue;
                }

                PuyotanMatch m = parent.match;
                int mask = m.getDecisionMask();
                if (mask == 0) {
                    mask = m.stepUntilDecision();
                    if (m.getStatus() != MatchStatus::Playing || mask == 0) {
                        AbsNode n = parent;
                        n.match = m;
                        // Terminal: use full eval (rare, cheap per-call)
                        n.score = evaluateAbsState(m, my_id, cfg.eval_weights);
                        local.push_back(n);
                        continue;
                    }
                }

                const int active_player = (mask & 1) ? 0 : 1;

                for (int act_idx = 0; act_idx < kNumRLActions; ++act_idx) {
                    PuyotanMatch child = m;
                    child.setAction(active_player, getRLAction(act_idx));
                    child.stepUntilDecision();

                    const AbsCategory cat = classifyMove(m, child, active_player);
                    // CHEAP proxy: parent's full eval + incremental game score
                    const float proxy = proxyScore(parent, child, my_id);
                    const int first_act = (step == 0) ? act_idx : parent.first_action;

                    local.push_back(AbsNode{child, proxy, first_act, cat});
                }
            }
        }

        // Merge thread-local candidates
        std::vector<AbsNode> candidates;
        {
            size_t total = 0;
            for (const auto& v : tl_candidates) total += v.size();
            candidates.reserve(total);
            for (auto& v : tl_candidates)
                for (auto& n : v) candidates.push_back(std::move(n));
        }
        if (candidates.empty()) break;

        // Determine whose turn this step is
        bool step_is_my_turn = true;
        for (const auto& n : candidates) {
            int msk = n.match.getDecisionMask();
            if (msk != 0) {
                step_is_my_turn = (((msk & 1) ? 0 : 1) == my_id);
                break;
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Prune to budget using cheap proxy score
        // ---------------------------------------------------------------
        const auto& budgets = step_is_my_turn ? cfg.my_budgets : cfg.opp_budgets;
        pruneByCategory(candidates, budgets, step_is_my_turn, current_beam);

        // ---------------------------------------------------------------
        // Phase 3: Full precision evaluation on SURVIVORS ONLY
        //          (SoloBeamEvaluator called at most budget times per step,
        //           vs. budget×22 times before. ~22x speedup.)
        // ---------------------------------------------------------------
        const int num_survivors = static_cast<int>(current_beam.size());
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < num_survivors; ++i) {
            auto& node = current_beam[i];
            if (node.match.getStatus() == MatchStatus::Playing) {
                node.score = evaluateAbsState(node.match, my_id, cfg.eval_weights);
            }
        }
    }

    if (current_beam.empty()) {
        res.best_eval = evaluateAbsState(match, my_id, cfg.eval_weights);
        return res;
    }

    // --- Select best action from final beam ---
    float best_eval = -1e9f;
    int best_act = current_beam[0].first_action;
    std::unordered_map<int, float> best_per_action;

    for (const auto& node : current_beam) {
        if (node.first_action < 0) continue;
        auto it = best_per_action.find(node.first_action);
        if (it == best_per_action.end() || node.score > it->second)
            best_per_action[node.first_action] = node.score;
        if (node.score > best_eval) {
            best_eval = node.score;
            best_act  = node.first_action;
        }
    }

    for (const auto& [act, val] : best_per_action)
        res.candidate_evals.emplace_back(act, val);

    res.best_action = (best_act >= 0) ? best_act : 0;
    res.best_eval   = (best_eval > -1e8f) ? best_eval
                                           : evaluateAbsState(match, my_id, cfg.eval_weights);
    return res;
}

} // namespace puyotan::search
