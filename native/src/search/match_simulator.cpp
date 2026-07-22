#include <puyotan/search/match_simulator.hpp>
#include <puyotan/engine/match.hpp>
#include <algorithm>
#include <omp.h>

namespace puyotan::search {

MatchResult simulateVsMatch(
    const VsBeamConfig& p1_cfg,
    const VsBeamConfig& p2_cfg,
    uint32_t seed,
    int max_frames
) noexcept {
    PuyotanMatch match(seed);
    match.start();

    // Local copies so we can update context each frame without mutating the const args.
    VsBeamConfig p1_cfg_local = p1_cfg;
    VsBeamConfig p2_cfg_local = p2_cfg;

    int max_chain_p1 = 0;
    int max_chain_p2 = 0;

    while (match.getStatus() == MatchStatus::Playing && match.getFrame() < max_frames) {
        int mask = match.getDecisionMask();
        if (mask != 0) {
            if (mask & 1) {
                const PuyotanPlayer& ep = match.getPlayer(1);
                const PuyotanPlayer& mp = match.getPlayer(0);
                VsEvalContext& ctx = p1_cfg_local.context;
                ctx.enemy_field            = ep.field;
                ctx.enemy_active_next_pos  = ep.active_next_pos;
                ctx.enemy_action_type      = ep.current_action.action.type;
                ctx.enemy_chain_count      = ep.chain_count;
                ctx.enemy_score            = ep.score;
                ctx.enemy_used_score       = ep.used_score;
                ctx.enemy_active_ojama     = ep.active_ojama;
                ctx.enemy_non_active_ojama = ep.non_active_ojama;
                ctx.my_active_ojama        = mp.active_ojama;
                ctx.my_non_active_ojama    = mp.non_active_ojama;
                auto action_pair = vsBeamSearch(mp, match.getTsumo(), p1_cfg_local);
                match.setAction(0, getRLAction(action_pair.first));
            }
            if (mask & 2) {
                const PuyotanPlayer& ep = match.getPlayer(0);
                const PuyotanPlayer& mp = match.getPlayer(1);
                VsEvalContext& ctx = p2_cfg_local.context;
                ctx.enemy_field            = ep.field;
                ctx.enemy_active_next_pos  = ep.active_next_pos;
                ctx.enemy_action_type      = ep.current_action.action.type;
                ctx.enemy_chain_count      = ep.chain_count;
                ctx.enemy_score            = ep.score;
                ctx.enemy_used_score       = ep.used_score;
                ctx.enemy_active_ojama     = ep.active_ojama;
                ctx.enemy_non_active_ojama = ep.non_active_ojama;
                ctx.my_active_ojama        = mp.active_ojama;
                ctx.my_non_active_ojama    = mp.non_active_ojama;
                auto action_pair = vsBeamSearch(mp, match.getTsumo(), p2_cfg_local);
                match.setAction(1, getRLAction(action_pair.first));
            }
        }

        match.stepNextFrame();

        max_chain_p1 = std::max<int>(max_chain_p1, match.getPlayer(0).chain_count);
        max_chain_p2 = std::max<int>(max_chain_p2, match.getPlayer(1).chain_count);
    }

    MatchResult res;
    res.status = match.getStatus();
    res.score_p1 = match.getPlayer(0).score;
    res.score_p2 = match.getPlayer(1).score;
    res.max_chain_p1 = max_chain_p1;
    res.max_chain_p2 = max_chain_p2;
    res.total_frames = match.getFrame();
    return res;
}

std::vector<MatchResult> simulateVsMatchesParallel(
    const VsBeamConfig& p1_cfg,
    const VsBeamConfig& p2_cfg,
    const std::vector<uint32_t>& seeds,
    int max_frames
) noexcept {
    const size_t num_games = seeds.size();
    std::vector<MatchResult> results(num_games);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(num_games); ++i) {
        results[i] = simulateVsMatch(p1_cfg, p2_cfg, seeds[i], max_frames);
    }

    return results;
}

} // namespace puyotan::search
