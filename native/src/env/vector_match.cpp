#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/reward.hpp>
#include <puyotan/env/vector_match.hpp>
#include <span>
namespace puyotan {
namespace {
// Terminated mapping: Any status other than Playing (1) is considered terminal.
static constexpr bool kTermTable[] = {
    true,  // Ready (0)
    false, // Playing (1)
    true,  // WinP1 (2)
    true,  // WinP2 (3)
    true   // Draw (4)
};
} // anonymous namespace
PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, uint32_t base_seed)
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
    for (int i = 0; i < num_matches; ++i) {
        matches_.emplace_back(base_seed + i);
        matches_.back().start();
        matches_.back().stepUntilDecision();
    }
}
void PuyotanVectorMatch::reset(int id) noexcept {
    if (id == -1) {
#pragma omp parallel for
        for (int i = 0; i < (int)matches_.size(); ++i) {
            matches_[i] = PuyotanMatch(base_seed_ + i);
            matches_[i].start();             // Fixed: Must call start()
            matches_[i].stepUntilDecision(); // Fixed: Advance to decision state
        }
    } else {
        matches_[id] = PuyotanMatch(base_seed_ + id);
        matches_[id].start();             // Fixed: Must call start()
        matches_[id].stepUntilDecision(); // Fixed: Advance to decision state
    }
}
std::vector<int> PuyotanVectorMatch::stepUntilDecision() {
    std::vector<int> masks(matches_.size());
#pragma omp parallel for
    for (int i = 0; i < (int)matches_.size(); ++i) {
        masks[i] = matches_[i].stepUntilDecision();
    }
    return masks;
}
void PuyotanVectorMatch::setActions(const std::vector<int>& match_indices,
                                    const std::vector<int>& player_ids,
                                    const std::vector<Action>& actions) {
    for (size_t i = 0; i < match_indices.size(); ++i) {
        matches_[match_indices[i]].setAction(player_ids[i], actions[i]);
    }
}
void PuyotanVectorMatch::stepNative(
    std::span<const int> p1_actions,
    std::span<const int> p2_actions,
    std::span<float> out_rewards,
    std::span<float> out_dones,
    std::span<int32_t> out_chains,
    std::span<int32_t> out_scores,
    std::span<uint8_t> out_obs) {
    const int n = static_cast<int>(matches_.size());
// Parallel simulation -- no GIL, no Python objects.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& m = matches_[i];
        const auto& p1_pre = m.getPlayer(0);
        const auto& p2_pre = m.getPlayer(1);
        int start_score_p1 = p1_pre.score;
        int start_score_p2 = p2_pre.score;
        int pre_ojama_p1 = p1_pre.total_ojama_dropped;
        int pre_ojama_p2 = p2_pre.total_ojama_dropped;
        m.setAction(0, getRLAction(p1_actions[i]));
        m.setAction(1, getRLAction(p2_actions[i]));
        m.stepUntilDecision();
        RewardContext ctx = reward_calc.extractContext(
            m, start_score_p1, start_score_p2, pre_ojama_p1, pre_ojama_p2);
        out_rewards[i] = reward_calc.calculate(ctx, 0);
        out_chains[i] = ctx.p1_chain_count;
        out_scores[i] = ctx.p1_delta_score;
        bool is_term = kTermTable[static_cast<uint8_t>(ctx.status)];
        out_dones[i] = is_term ? 1.0f : 0.0f;
        if (is_term) {
            m = PuyotanMatch(base_seed_ + i);
            m.start();
            m.stepUntilDecision();
        }
        // Write observation directly into the output span.
        uint8_t* obs_ptr = out_obs.data() + i * ObservationBuilder::kBytesPerObservation;
        ObservationBuilder::buildObservation(m, obs_ptr);
    }
}
void PuyotanVectorMatch::getObservationsNative(std::span<uint8_t> out_obs) const {
    const int n = static_cast<int>(matches_.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        uint8_t* obs_ptr = out_obs.data() + i * ObservationBuilder::kBytesPerObservation;
        ObservationBuilder::buildObservation(matches_[i], obs_ptr);
    }
}
} // namespace puyotan
