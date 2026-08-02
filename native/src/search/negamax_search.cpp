#include <puyotan/search/negamax_search.hpp>
#include <algorithm>
#include <limits>

namespace puyotan::search {

float evaluateMatchState(const PuyotanMatch& match, int my_id, const VsBeamEvalWeights& weights) noexcept {
    const MatchStatus st = match.getStatus();
    if (st == MatchStatus::WinP1) {
        return my_id == 0 ? 1000000.0f : -1000000.0f;
    }
    if (st == MatchStatus::WinP2) {
        return my_id == 1 ? 1000000.0f : -1000000.0f;
    }
    if (st == MatchStatus::Draw) {
        return 0.0f;
    }

    const auto& my_p  = match.getPlayer(my_id);
    const auto& opp_p = match.getPlayer(1 - my_id);

    VsEvalContext my_ctx;
    my_ctx.enemy_field            = opp_p.field;
    my_ctx.enemy_active_next_pos  = opp_p.active_next_pos;
    my_ctx.enemy_action_type      = opp_p.current_action.action.type;
    my_ctx.enemy_chain_count      = opp_p.chain_count;
    my_ctx.enemy_score            = opp_p.score;
    my_ctx.enemy_used_score       = opp_p.used_score;
    my_ctx.enemy_active_ojama     = opp_p.active_ojama;
    my_ctx.enemy_non_active_ojama = opp_p.non_active_ojama;
    my_ctx.my_active_ojama        = my_p.active_ojama;
    my_ctx.my_non_active_ojama    = my_p.non_active_ojama;

    VsEvalContext opp_ctx;
    opp_ctx.enemy_field            = my_p.field;
    opp_ctx.enemy_active_next_pos  = my_p.active_next_pos;
    opp_ctx.enemy_action_type      = my_p.current_action.action.type;
    opp_ctx.enemy_chain_count      = my_p.chain_count;
    opp_ctx.enemy_score            = my_p.score;
    opp_ctx.enemy_used_score       = my_p.used_score;
    opp_ctx.enemy_active_ojama     = my_p.active_ojama;
    opp_ctx.enemy_non_active_ojama = my_p.non_active_ojama;
    opp_ctx.my_active_ojama        = opp_p.active_ojama;
    opp_ctx.my_non_active_ojama    = opp_p.non_active_ojama;

    const float my_pot  = VsBeamEvaluator::evaluate(my_p.field,  weights, &my_ctx);
    const float opp_pot = VsBeamEvaluator::evaluate(opp_p.field, weights, &opp_ctx);

    const float my_total  = static_cast<float>(my_p.score)  + my_pot;
    const float opp_total = static_cast<float>(opp_p.score) + opp_pot;

    return my_total - opp_total;
}

static float negamaxRec(PuyotanMatch match, int my_id, int depth, float alpha, float beta, const NegamaxConfig& cfg) noexcept {
    if (match.getStatus() != MatchStatus::Playing || depth <= 0) {
        return evaluateMatchState(match, my_id, cfg.vs_config.eval_weights);
    }

    int mask = match.getDecisionMask();
    bool is_post_chain_cutoff = false;

    if (mask == 0) {
        int next_mask = match.stepUntilDecision();
        if (match.getStatus() != MatchStatus::Playing || next_mask == 0 || depth <= 0) {
            return evaluateMatchState(match, my_id, cfg.vs_config.eval_weights);
        }
        mask = next_mask;
        if (cfg.chain_cutoff_enabled) {
            is_post_chain_cutoff = true;
        }
    }

    // Determine current turn player (0 or 1)
    const int current_player = (mask & 1) ? 0 : 1;
    const bool is_my_turn = (current_player == my_id);

    const PuyotanPlayer& curr_p = match.getPlayer(current_player);
    const PuyotanPlayer& opp_p = match.getPlayer(1 - current_player);

    // Interior nodes use a lightweight config to avoid expensive nested beam searches.
    const VsBeamConfig& base_cfg = (cfg.use_interior_config) ? cfg.interior_vs_config : cfg.vs_config;
    VsBeamConfig player_cfg = base_cfg;
    VsEvalContext& ctx = player_cfg.context;
    ctx.enemy_field            = opp_p.field;
    ctx.enemy_active_next_pos  = opp_p.active_next_pos;
    ctx.enemy_action_type      = opp_p.current_action.action.type;
    ctx.enemy_chain_count      = opp_p.chain_count;
    ctx.enemy_score            = opp_p.score;
    ctx.enemy_used_score       = opp_p.used_score;
    ctx.enemy_active_ojama     = opp_p.active_ojama;
    ctx.enemy_non_active_ojama = opp_p.non_active_ojama;
    ctx.my_active_ojama        = curr_p.active_ojama;
    ctx.my_non_active_ojama    = curr_p.non_active_ojama;

    const int target_candidate_n = cfg.interior_candidate_n > 0 ? cfg.interior_candidate_n : cfg.candidate_n;
    auto candidates = vsBeamSearchTopN(curr_p, match.getTsumo(), player_cfg, target_candidate_n);
    if (candidates.empty()) {
        return evaluateMatchState(match, my_id, cfg.vs_config.eval_weights);
    }

    const int next_depth = is_post_chain_cutoff ? std::min(depth - 1, 1) : (depth - 1);

    if (is_my_turn) {
        float max_eval = -1e9f;
        for (const auto& [act_idx, _] : candidates) {
            PuyotanMatch next_match = match;
            next_match.setAction(current_player, getRLAction(act_idx));
            next_match.stepUntilDecision();

            float val = negamaxRec(next_match, my_id, next_depth, alpha, beta, cfg);
            max_eval = std::max(max_eval, val);
            alpha = std::max(alpha, val);
            if (beta <= alpha) {
                break; // Alpha-beta cut
            }
        }
        return max_eval;
    } else {
        float min_eval = 1e9f;
        for (const auto& [act_idx, _] : candidates) {
            PuyotanMatch next_match = match;
            next_match.setAction(current_player, getRLAction(act_idx));
            next_match.stepUntilDecision();

            float val = negamaxRec(next_match, my_id, next_depth, alpha, beta, cfg);
            min_eval = std::min(min_eval, val);
            beta = std::min(beta, val);
            if (beta <= alpha) {
                break; // Alpha-beta cut
            }
        }
        return min_eval;
    }
}

#include <omp.h>

NegamaxResult negamaxSearch(const PuyotanMatch& match, int my_id, const NegamaxConfig& cfg) noexcept {
    NegamaxResult res;
    if (match.getStatus() != MatchStatus::Playing) {
        res.best_eval = evaluateMatchState(match, my_id, cfg.vs_config.eval_weights);
        return res;
    }

    const PuyotanPlayer& my_p = match.getPlayer(my_id);
    const PuyotanPlayer& opp_p = match.getPlayer(1 - my_id);

    VsBeamConfig player_cfg = cfg.vs_config;
    VsEvalContext& ctx = player_cfg.context;
    ctx.enemy_field            = opp_p.field;
    ctx.enemy_active_next_pos  = opp_p.active_next_pos;
    ctx.enemy_action_type      = opp_p.current_action.action.type;
    ctx.enemy_chain_count      = opp_p.chain_count;
    ctx.enemy_score            = opp_p.score;
    ctx.enemy_used_score       = opp_p.used_score;
    ctx.enemy_active_ojama     = opp_p.active_ojama;
    ctx.enemy_non_active_ojama = opp_p.non_active_ojama;
    ctx.my_active_ojama        = my_p.active_ojama;
    ctx.my_non_active_ojama    = my_p.non_active_ojama;

    auto candidates = vsBeamSearchTopN(my_p, match.getTsumo(), player_cfg, cfg.candidate_n);
    if (candidates.empty()) {
        res.best_eval = evaluateMatchState(match, my_id, cfg.vs_config.eval_weights);
        return res;
    }

    const int num_candidates = static_cast<int>(candidates.size());
    
    // Allocate 64-byte aligned structure per thread to prevent False Sharing (CPU cache line interference)
    struct alignas(64) ThreadResult {
        std::vector<std::pair<int, float>> items;
    };

    // Up to 2 thread private buffers for AMD 3020e
    std::vector<ThreadResult> thread_buffers(2);

    #pragma omp parallel num_threads(2)
    {
        const int tid = omp_get_thread_num();
        auto& local_buf = thread_buffers[tid].items;

        #pragma omp for schedule(static)
        for (int i = 0; i < num_candidates; ++i) {
            const int act_idx = candidates[i].first;
            PuyotanMatch next_match = match; // Pure stack value copy (Zero shared memory)
            next_match.setAction(my_id, getRLAction(act_idx));
            next_match.stepUntilDecision();

            float eval_val = negamaxRec(next_match, my_id, cfg.depth - 1, -1e9f, 1e9f, cfg);
            local_buf.emplace_back(act_idx, eval_val);
        }
    }

    // Merge results seamlessly without any locks
    float best_eval = -1e9f;
    int best_action = candidates[0].first;

    for (const auto& buf : thread_buffers) {
        for (const auto& [act_idx, eval_val] : buf.items) {
            res.candidate_evals.emplace_back(act_idx, eval_val);
            if (eval_val > best_eval) {
                best_eval = eval_val;
                best_action = act_idx;
            }
        }
    }

    res.best_action = best_action;
    res.best_eval = best_eval;
    return res;
}

} // namespace puyotan::search
