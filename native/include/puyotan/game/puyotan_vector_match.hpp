#pragma once

#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <puyotan/game/puyotan_match.hpp>

namespace puyotan {

/**
 * PuyotanVectorMatch
 *   Manages a collection of PuyotanMatch instances for high-throughput batch simulation.
 *   Optimized for RL training loops where multiple agents act in parallel.
 */
class PuyotanVectorMatch {
public:
    explicit PuyotanVectorMatch(int num_matches, int32_t base_seed = 0);

    /**
     * Resets one or all matches.
     */
    void reset(int id = -1);

    /**
     * Runs all matches until they require decision (Step until decision).
     * @return Array of masks [N] where mask & (1 << player_id) means decision needed.
     */
    std::vector<int> step_until_decision();

    /**
     * Batch-set actions for multiple players across matches.
     */
    void set_actions(const std::vector<int>& match_indices, 
                     const std::vector<int>& player_ids,
                     const std::vector<Action>& actions);

    /**
     * Bulk step function incorporating action decoding, stepping, auto-reset, and reward calculation (OpenMP accelerated).
     * @param p1_actions Action indices [0-21] for Player 1.
     * @param p2_actions Action indices [0-21] for Player 2. If empty, P2 passes.
     * @return A tuple of (observations, rewards, terminated).
     */
    pybind11::tuple step(pybind11::array_t<int> p1_actions, std::optional<pybind11::array_t<int>> p2_actions);

    /**
     * Bulk observation generation.
     * @return Flat array of [N, 2, 5, 6, 13] representing all fields.
     */
    pybind11::array_t<uint8_t> get_observations_all() const;

    size_t size() const { return matches_.size(); }
    PuyotanMatch& get_match(int i) { return *matches_[i]; }

private:
    std::vector<std::unique_ptr<PuyotanMatch>> matches_;
    int32_t base_seed_;
    
    // For reward calculation caching
    std::vector<int> prev_scores_;
    std::vector<int> prev_ojama_;
};

} // namespace puyotan
