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
    : base_seed_(base_seed), env_seeds_(num_matches), episode_steps_(num_matches, 0) {
    matches_.reserve(num_matches);
    uint32_t seed = base_seed;
    for (int i = 0; i < num_matches; ++i) {
        // Xorshift32 to diversify initial seeds across environments
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        if (seed == 0) seed = base_seed + i + 1;
        env_seeds_[i] = seed;
        matches_.emplace_back(env_seeds_[i]);
        matches_.back().start();
        matches_.back().stepUntilDecision();
    }
}
void PuyotanVectorMatch::reset(int id) noexcept {
    if (id == -1) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)matches_.size(); ++i) {
            uint32_t seed = env_seeds_[i];
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            if (seed == 0) seed = base_seed_ + i + 1; // fallback
            env_seeds_[i] = seed;
            matches_[i] = PuyotanMatch(seed);
            matches_[i].start();
            matches_[i].stepUntilDecision();
        }
    } else {
        uint32_t seed = env_seeds_[id];
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        if (seed == 0) seed = base_seed_ + id + 1; // fallback
        env_seeds_[id] = seed;
        matches_[id] = PuyotanMatch(seed);
        matches_[id].start();
        matches_[id].stepUntilDecision();
        episode_steps_[id] = 0;
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
    std::span<const int8_t> p1_actions,
    std::span<const int8_t> p2_actions,
    std::span<float> out_rewards,
    std::span<float> out_dones,
    std::span<int8_t> out_chains,
    std::span<int32_t> out_scores,
    std::span<int8_t> out_potentials,
    std::span<uint8_t> out_obs) noexcept {
    const int n = static_cast<int>(matches_.size());
// Parallel simulation -- no GIL, no Python objects.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& m = matches_[i];
        
        // Take copies of pre-step player states
        PuyotanPlayer p1_pre = m.getPlayer(0);
        PuyotanPlayer p2_pre = m.getPlayer(1);

        m.setAction(0, getRLAction(p1_actions[i]));
        m.setAction(1, getRLAction(p2_actions[i]));

        int p1_max_chain = 0;
        int p2_max_chain = 0;
        int p1_ojama_dropped = 0;
        int p2_ojama_dropped = 0;

        // Drive stepNextFrame manually to inspect state transitions for metrics
        while (m.getStatus() == MatchStatus::Playing) {
            // Check if we reached decision point
            if (m.getPlayer(0).current_action.action.type == ActionType::None ||
                m.getPlayer(1).current_action.action.type == ActionType::None) {
                break;
            }

            const auto& p1_curr = m.getPlayer(0);
            const auto& p2_curr = m.getPlayer(1);

            // Track max chain achieved: sample BEFORE stepNextFrame clears chain_count on termination
            if (p1_curr.current_action.action.type == ActionType::Chain) {
                p1_max_chain = std::max(p1_max_chain, static_cast<int>(p1_curr.chain_count + 1));
            }
            if (p2_curr.current_action.action.type == ActionType::Chain) {
                p2_max_chain = std::max(p2_max_chain, static_cast<int>(p2_curr.chain_count + 1));
            }

            // Track ojama drops right before they execute
            if (p1_curr.current_action.action.type == ActionType::Ojama && p1_curr.current_action.remaining_frame == 0) {
                p1_ojama_dropped += std::min(static_cast<int>(p1_curr.active_ojama), config::Rule::kMaxOjamaPerFall);
            }
            if (p2_curr.current_action.action.type == ActionType::Ojama && p2_curr.current_action.remaining_frame == 0) {
                p2_ojama_dropped += std::min(static_cast<int>(p2_curr.active_ojama), config::Rule::kMaxOjamaPerFall);
            }

            m.stepNextFrame();
        }

        RewardContext ctx = reward_calc.extractContext(
            m, p1_pre, p2_pre, p1_ojama_dropped, p2_ojama_dropped, p1_max_chain, p2_max_chain);
        out_rewards[i] = reward_calc.calculate(ctx, 0);
        out_chains[i] = ctx.p1_chain_count;
        out_scores[i] = ctx.p1_delta_score;
        out_potentials[i] = static_cast<int8_t>(ctx.p1_potential_chain);
        bool is_term = kTermTable[static_cast<uint8_t>(ctx.status)];
        // Max episode steps: force truncation if limit is set and exceeded
        episode_steps_[i]++;
        if (!is_term && max_episode_steps_ > 0 && episode_steps_[i] >= max_episode_steps_) {
            is_term = true;
        }
        out_dones[i] = is_term ? 1.0f : 0.0f;
        if (is_term) {
            episode_steps_[i] = 0;
            uint32_t seed = env_seeds_[i];
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            if (seed == 0) seed = base_seed_ + i + 1;
            env_seeds_[i] = seed;
            m = PuyotanMatch(seed);
            m.start();
            m.stepUntilDecision();
        }
        // Write observation directly into the output span.
        uint8_t* obs_ptr = out_obs.data() + i * ObservationBuilder::kBytesPerObservation;
        ObservationBuilder::buildObservation(m, obs_ptr);
    }
}
void PuyotanVectorMatch::getObservationsNative(std::span<uint8_t> out_obs) const noexcept {
    const int n = static_cast<int>(matches_.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        uint8_t* obs_ptr = out_obs.data() + i * ObservationBuilder::kBytesPerObservation;
        ObservationBuilder::buildObservation(matches_[i], obs_ptr);
    }
}
} // namespace puyotan
