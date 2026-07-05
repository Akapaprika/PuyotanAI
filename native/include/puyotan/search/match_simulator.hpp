#pragma once
#include <vector>
#include <cstdint>
#include <puyotan/common/types.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {

struct MatchResult {
    MatchStatus status;
    int score_p1 = 0;
    int score_p2 = 0;
    int max_chain_p1 = 0;
    int max_chain_p2 = 0;
    int total_frames = 0;
};

MatchResult simulateVsMatch(
    const VsBeamConfig& p1_cfg,
    const VsBeamConfig& p2_cfg,
    uint32_t seed,
    int max_frames = 15000
) noexcept;

std::vector<MatchResult> simulateVsMatchesParallel(
    const VsBeamConfig& p1_cfg,
    const VsBeamConfig& p2_cfg,
    const std::vector<uint32_t>& seeds,
    int max_frames = 15000
) noexcept;

} // namespace puyotan::search
