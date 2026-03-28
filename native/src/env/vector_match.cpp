#include <puyotan/env/vector_match.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <algorithm>
#include <cstring>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/reward.hpp>
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
    if (idx < 0 || idx >= 22) return Action{ActionType::PASS};
    if (idx < 6) return Action{ActionType::PUT, static_cast<int8_t>(idx), Rotation::Up};
    if (idx < 11) return Action{ActionType::PUT, static_cast<int8_t>(idx - 6), Rotation::Right}; // Col 0-4
    if (idx < 17) return Action{ActionType::PUT, static_cast<int8_t>(idx - 11), Rotation::Down}; // Col 0-5
    return Action{ActionType::PUT, static_cast<int8_t>(idx - 16), Rotation::Left}; // Col 1-5
}
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
                                         std::optional<pybind11::array_t<int>> p2_actions,
                                         std::optional<pybind11::array_t<uint8_t>> out_obs) {
    const int n = static_cast<int>(matches_.size());
    auto p1_ptr = p1_actions.data();
    auto p2_ptr = p2_actions.has_value() ? p2_actions->data() : nullptr;

    pybind11::array_t<float> rewards(n);
    pybind11::array_t<bool> terminated(n);
    pybind11::array_t<int> chains(n);
    float* rew_ptr = static_cast<float*>(rewards.mutable_data());
    bool* term_ptr = static_cast<bool*>(terminated.mutable_data());
    int* chain_ptr = static_cast<int*>(chains.mutable_data());

    {
        pybind11::gil_scoped_release release;
        static constexpr bool kTermTable[] = {true, false, true, true, true};
        RewardCalculator reward_calc;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            auto& m = matches_[i];
            
            int start_score_p1 = m.getPlayer(0).score;

            m.setAction(0, GET_ACTION(p1_ptr[i]));
            if (p2_ptr) m.setAction(1, GET_ACTION(p2_ptr[i]));
            else m.setAction(1, Action{ActionType::PASS});

            while (m.getStatus() == MatchStatus::PLAYING) {
                int mask = m.stepUntilDecision();
                if (mask == 3 || mask == 0) break;
                if ((mask & 1) != 0) m.setAction(0, Action{ActionType::PASS});
                if ((mask & 2) != 0) m.setAction(1, Action{ActionType::PASS});
            }

            const auto& p1 = m.getPlayer(0);
            MatchStatus status = m.getStatus();
            
            TurnStats stats { p1.score - start_score_p1, p1.last_chain_count };
            rew_ptr[i] = reward_calc.calculate(stats, status, 0);
            chain_ptr[i] = p1.last_chain_count;

            bool is_term = kTermTable[static_cast<uint8_t>(status)];
            term_ptr[i] = is_term;

            if (is_term) {
                m = PuyotanMatch(base_seed_ + i);
                m.start();
                m.stepUntilDecision();
            }
        }
    }
    return pybind11::make_tuple(getObservationsAll(out_obs), std::move(rewards), std::move(terminated), std::move(chains));
}

pybind11::array_t<uint8_t> PuyotanVectorMatch::getObservationsAll(std::optional<pybind11::array_t<uint8_t>> out_obs) const {
    const int n = static_cast<int>(matches_.size());
    static constexpr std::size_t kBytesPerCol = 14; 
    static constexpr std::size_t kBytesPerColor = 6 * 14;
    static constexpr std::size_t kBytesPerField = 5 * 6 * 14;
    static constexpr std::size_t kBytesPerObservation = 2 * kBytesPerField;

    pybind11::array_t<uint8_t> arr;
    if (out_obs.has_value()) arr = *out_obs;
    else arr = pybind11::array_t<uint8_t>({(std::size_t)n, (std::size_t)2, (std::size_t)5, (std::size_t)6, (std::size_t)14});
    uint8_t* out_base = static_cast<uint8_t*>(arr.mutable_data());

    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* obs_ptr = out_base + i * ObservationBuilder::kBytesPerObservation;
            ObservationBuilder::build_observation(matches_[i], obs_ptr);
        }
    }
    return arr;
}

} // namespace puyotan
