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

    int max_chain_p1 = 0;
    int max_chain_p2 = 0;

    while (match.getStatus() == MatchStatus::Playing && match.getFrame() < max_frames) {
        int mask = match.getDecisionMask();
        if (mask != 0) {
            if (mask & 1) {
                auto action_pair = vsBeamSearch(match.getPlayer(0), match.getTsumo(), p1_cfg);
                match.setAction(0, getRLAction(action_pair.first));
            }
            if (mask & 2) {
                auto action_pair = vsBeamSearch(match.getPlayer(1), match.getTsumo(), p2_cfg);
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
