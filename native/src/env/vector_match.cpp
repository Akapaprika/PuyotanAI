#include <puyotan/env/vector_match.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/reward.hpp>
#include <puyotan/core/gravity.hpp>

#include <algorithm>
#include <cstring>
#include <span>
#include <immintrin.h>

namespace puyotan {

namespace {

/**
 * Maps RL Action Index [0-21] to game Action [Column, Rotation].
 * Layout:
 *   [0-5]   -> Up, Col 0-5
 *   [6-11]  -> Right, Col 0-5
 *   [12-17] -> Down, Col 0-5
 *   [18-21] -> Left, Col 0-3 (Columns 4,5 are restricted in Left rotation to prevent wall-clip errors)
 */
Action GET_ACTION(int idx) {
    constexpr int w = config::Board::kWidth;
    constexpr int max_actions = w + (w - 1) + w + (w - 1); // 22 actions
    
    if (idx < 0 || idx >= max_actions) return Action{ActionType::Pass};
    if (idx < w) return Action{ActionType::Put, static_cast<int8_t>(idx), Rotation::Up};
    if (idx < w + (w - 1)) return Action{ActionType::Put, static_cast<int8_t>(idx - w), Rotation::Right}; // Col 0-4
    if (idx < w + (w - 1) + w) return Action{ActionType::Put, static_cast<int8_t>(idx - (w + w - 1)), Rotation::Down}; // Col 0-5
    return Action{ActionType::Put, static_cast<int8_t>(idx - (w + w - 1 + w)), Rotation::Left}; // Col 1-5
}

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
            matches_[i].start(); // Fixed: Must call start()
            matches_[i].stepUntilDecision(); // Fixed: Advance to decision state
        }
    } else {
        matches_[id] = PuyotanMatch(base_seed_ + id);
        matches_[id].start(); // Fixed: Must call start()
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

pybind11::tuple PuyotanVectorMatch::step(pybind11::array_t<int> p1_actions, 
                                         pybind11::array_t<int> p2_actions,
                                         std::optional<pybind11::array_t<uint8_t>> out_obs) {
    const int n = static_cast<int>(matches_.size());
    auto p1_ptr = p1_actions.data();
    auto p2_ptr = p2_actions.data();

    pybind11::array_t<float> rewards(n);
    pybind11::array_t<bool> terminated(n);
    pybind11::array_t<int> chains(n);
    pybind11::array_t<int> scores(n);
    float* rew_ptr = static_cast<float*>(rewards.mutable_data());
    bool* term_ptr = static_cast<bool*>(terminated.mutable_data());
    int* chain_ptr = static_cast<int*>(chains.mutable_data());
    int* score_ptr = static_cast<int*>(scores.mutable_data());

    {
        pybind11::gil_scoped_release release;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            auto& m = matches_[i];
            
            const auto& p1_pre = m.getPlayer(0);
            const auto& p2_pre = m.getPlayer(1);
            int start_score_p1 = p1_pre.score;
            int start_score_p2 = p2_pre.score;
            int pre_ojama_p1 = p1_pre.total_ojama_dropped;
            int pre_ojama_p2 = p2_pre.total_ojama_dropped;

            m.setAction(0, GET_ACTION(p1_ptr[i]));
            m.setAction(1, GET_ACTION(p2_ptr[i]));

            // Advance engine
            m.stepUntilDecision();

            RewardContext ctx = reward_calc.extractContext(m, start_score_p1, start_score_p2, pre_ojama_p1, pre_ojama_p2);

            rew_ptr[i] = reward_calc.calculate(ctx, 0);
            chain_ptr[i] = ctx.p1_chain_count;
            score_ptr[i] = ctx.p1_delta_score;

            bool is_term = kTermTable[static_cast<uint8_t>(ctx.status)];
            term_ptr[i] = is_term;

            if (is_term) {
                m = PuyotanMatch(base_seed_ + i);
                m.start();
                m.stepUntilDecision();
            }
        }
    }
    return pybind11::make_tuple(getObservationsAll(out_obs), std::move(rewards), std::move(terminated), std::move(chains), std::move(scores));
}

pybind11::array_t<uint8_t> PuyotanVectorMatch::getObservationsAll(std::optional<pybind11::array_t<uint8_t>> out_obs) const {
    const int n = static_cast<int>(matches_.size());
    static constexpr std::size_t kBytesPerCol = config::Board::kObsHeight; 
    static constexpr std::size_t kBytesPerColor = config::Board::kWidth * kBytesPerCol;
    static constexpr std::size_t kBytesPerField = config::Board::kNumColors * kBytesPerColor;
    static constexpr std::size_t kBytesPerObservation = config::Rule::kNumPlayers * kBytesPerField;

    pybind11::array_t<uint8_t> arr;
    if (out_obs.has_value()) arr = *out_obs;
    else arr = pybind11::array_t<uint8_t>({
        (std::size_t)n, 
        (std::size_t)config::Rule::kNumPlayers, 
        (std::size_t)config::Board::kNumColors, 
        (std::size_t)config::Board::kWidth, 
        (std::size_t)config::Board::kObsHeight
    });
    uint8_t* out_base = static_cast<uint8_t*>(arr.mutable_data());

    {
        // RELEASE GIL: Batch observation construction is also parallelized via OpenMP.
        pybind11::gil_scoped_release release;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* obs_ptr = out_base + i * ObservationBuilder::kBytesPerObservation;
            ObservationBuilder::buildObservation(matches_[i], obs_ptr);
        }
    }
    return arr;
}
void PuyotanVectorMatch::stepNative(
    std::span<const int>   p1_actions,
    std::span<const int>   p2_actions,
    std::span<float>       out_rewards,
    std::span<float>       out_dones,
    std::span<int32_t>     out_chains,
    std::span<int32_t>     out_scores,
    std::span<uint8_t>     out_obs)
{
    const int n = static_cast<int>(matches_.size());

    // Parallel simulation — no GIL, no Python objects.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& m = matches_[i];

        const auto& p1_pre = m.getPlayer(0);
        const auto& p2_pre = m.getPlayer(1);
        int start_score_p1 = p1_pre.score;
        int start_score_p2 = p2_pre.score;
        int pre_ojama_p1   = p1_pre.total_ojama_dropped;
        int pre_ojama_p2   = p2_pre.total_ojama_dropped;

        m.setAction(0, GET_ACTION(p1_actions[i]));
        m.setAction(1, GET_ACTION(p2_actions[i]));
        m.stepUntilDecision();

        RewardContext ctx = reward_calc.extractContext(
            m, start_score_p1, start_score_p2, pre_ojama_p1, pre_ojama_p2);

        out_rewards[i] = reward_calc.calculate(ctx, 0);
        out_chains[i]  = ctx.p1_chain_count;
        out_scores[i]  = ctx.p1_delta_score;

        bool is_term   = kTermTable[static_cast<uint8_t>(ctx.status)];
        out_dones[i]   = is_term ? 1.0f : 0.0f;

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
