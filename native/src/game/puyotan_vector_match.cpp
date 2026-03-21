#include <puyotan/game/puyotan_vector_match.hpp>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstddef>

namespace puyotan {

PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, int32_t base_seed) 
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
    for (int i = 0; i < num_matches; ++i) {
        matches_.emplace_back(std::make_unique<PuyotanMatch>(base_seed + i));
    }
}

void PuyotanVectorMatch::reset(int id) {
    if (id >= 0) {
        matches_[id] = std::make_unique<PuyotanMatch>(base_seed_ + id);
    } else {
        const int n = static_cast<int>(matches_.size());
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            matches_[i] = std::make_unique<PuyotanMatch>(base_seed_ + i);
        }
    }
}

std::vector<int> PuyotanVectorMatch::step_until_decision() {
    const int n = static_cast<int>(matches_.size());
    std::vector<int> masks(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masks[i] = matches_[i]->stepUntilDecision();
    }
    return masks;
}

void PuyotanVectorMatch::set_actions(const std::vector<int>& match_indices, 
                                    const std::vector<int>& player_ids,
                                    const std::vector<Action>& actions) {
    const int n = static_cast<int>(match_indices.size());
    // This is typically called from Python and is not heavily parallelizable 
    // unless we have a large number of actions to set.
    for (int i = 0; i < n; ++i) {
        matches_[match_indices[i]]->setAction(player_ids[i], actions[i]);
    }
}

pybind11::array_t<float> PuyotanVectorMatch::get_observations_all() const {
    const int n = static_cast<int>(matches_.size());
    const std::size_t colors = config::Board::kNumColors;
    const std::size_t width  = config::Board::kWidth;
    const std::size_t height = config::Board::kHeight;

    // 1. Allocate array (Must hold GIL)
    pybind11::array_t<float> arr({(std::size_t)n, (std::size_t)2, colors, width, height});
    auto r = arr.mutable_unchecked<5>();

    // 2. Parallel processing (Release GIL)
    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int p_idx = 0; p_idx < 2; ++p_idx) {
                const auto& player = matches_[i]->getPlayer(p_idx);
                for (int c = 0; c < (int)colors; ++c) {
                    auto bb = player.field.getBitboard(static_cast<Cell>(c));
                    for (int x = 0; x < (int)width; ++x) {
                        for (int y = 0; y < (int)height; ++y) {
                            r(i, p_idx, c, x, y) = bb.get(x, y) ? 1.0f : 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    return arr;
}

} // namespace puyotan
