#pragma once

#include <optional>
#include <span>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <puyotan/engine/match.hpp>
#include <puyotan/env/reward.hpp>

namespace puyotan {

/**
 * @class PuyotanVectorMatch
 * @brief Synchronous parallel orchestrator for batch Puyo Puyo matches.
 */
class PuyotanVectorMatch {
public:
    explicit PuyotanVectorMatch(int num_matches, uint32_t base_seed = 1u);

    void reset(int id = -1) noexcept;
    std::vector<int> stepUntilDecision();
    void setActions(const std::vector<int>& match_indices,
                     const std::vector<int>& player_ids,
                     const std::vector<Action>& actions);

    /**
     * @brief Python-facing step (used by GUI / external scripts).
     * Returns a pybind11::tuple (observations, rewards, terminated, chains, scores).
     */
    pybind11::tuple step(pybind11::array_t<int> p1_actions,
                         pybind11::array_t<int> p2_actions,
                         std::optional<pybind11::array_t<uint8_t>> out_obs = std::nullopt);

    /**
     * @brief High-performance native step for the C++ training loop.
     *
     * Bypasses all pybind11 overhead (no GIL, no Python object creation).
     * All output buffers must be pre-allocated by the caller.
     *
     * @param p1_actions  Action indices for P1 [n]
     * @param p2_actions  Action indices for P2 [n]
     * @param out_rewards Output float32 rewards [n]
     * @param out_dones   Output float32 terminated flags [n]
     * @param out_chains  Output int32 chain counts [n]
     * @param out_scores  Output int32 delta scores [n]
     * @param out_obs     Output uint8 observations [n * obs_bytes]
     */
    void stepNative(std::span<const int>   p1_actions,
                    std::span<const int>   p2_actions,
                    std::span<float>       out_rewards,
                    std::span<float>       out_dones,
                    std::span<int32_t>     out_chains,
                    std::span<int32_t>     out_scores,
                    std::span<uint8_t>     out_obs);

    /**
     * @brief Write observations to a pre-allocated uint8 buffer (no Python objects).
     * @param out_obs Buffer of size [n * kBytesPerObservation].
     */
    void getObservationsNative(std::span<uint8_t> out_obs) const;

    pybind11::array_t<uint8_t> getObservationsAll(
        std::optional<pybind11::array_t<uint8_t>> out_obs = std::nullopt) const;

    size_t size() const { return matches_.size(); }
    PuyotanMatch& getMatch(int i) { return matches_[i]; }
    const PuyotanMatch& getMatch(int i) const { return matches_[i]; }

    RewardCalculator reward_calc;

private:
    std::vector<PuyotanMatch> matches_;
    uint32_t base_seed_;
};

} // namespace puyotan
