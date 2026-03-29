#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <puyotan/engine/match.hpp>
#include <puyotan/env/reward.hpp>

namespace puyotan {

/**
 * @class PuyotanVectorMatch
 * @brief Synchronous parallel orchestrator for batch Puyo Puyo matches.
 * 
 * Optimized for Reinforcement Learning (RL) training loops. Manages multiple 
 * PuyotanMatch instances and provides high-throughput vectorized step/reset APIs.
 */
class PuyotanVectorMatch {
public:
    explicit PuyotanVectorMatch(int num_matches, uint32_t base_seed = 1u);

    /**
     * @brief Resets specific matches to their initial state.
     * @param id The match index (0 to N-1). If -1, all matches are reset in parallel.
     */
    void reset(int id = -1) noexcept;

    /**
     * @brief Steps all active matches until they reach a decision point (PUT).
     * @return A vector of bitmasks indicating which players need actions.
     */
    std::vector<int> stepUntilDecision();

    /**
     * Batch-set actions for multiple players across matches.
     */
    void setActions(const std::vector<int>& match_indices, 
                     const std::vector<int>& player_ids,
                     const std::vector<Action>& actions);

    /**
     * @brief High-performance batched simulation step for RL environments.
     * 
     * Incorporates action decoding, frame advancement, automatic resets, and 
     * reward calculation in a single OpenMP-accelerated call.
     * 
     * @param p1_actions Action indices for Player 1 (batch_size).
     * @param p2_actions Action indices for Player 2 (batch_size). Mirror P1 for solo mode.
     * @param out_obs Optional pre-allocated buffer for observation tensors.
     * @return py::tuple (observations, rewards, terminated, info).
     */
    pybind11::tuple step(pybind11::array_t<int> p1_actions, 
                         pybind11::array_t<int> p2_actions, 
                         std::optional<pybind11::array_t<uint8_t>> out_obs = std::nullopt);

    /**
     * @brief Generates observations for all active matches.
     * @param out_obs Optional pre-allocated buffer.
     * @return numpy array of shape [N, OBS_SIZE] uint8.
     */
    pybind11::array_t<uint8_t> getObservationsAll(std::optional<pybind11::array_t<uint8_t>> out_obs = std::nullopt) const;

    size_t size() const { return matches_.size(); }
    PuyotanMatch& getMatch(int i) { return matches_[i]; }
    const PuyotanMatch& getMatch(int i) const { return matches_[i]; }

    RewardCalculator reward_calc; ///< Configurable reward parameters

private:
    std::vector<PuyotanMatch> matches_;
    uint32_t base_seed_;
};

} // namespace puyotan
