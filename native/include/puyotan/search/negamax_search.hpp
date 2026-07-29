#pragma once

#include <utility>
#include <vector>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {

/**
 * @struct NegamaxConfig
 * @brief Configuration parameters for Negamax (adversarial lookahead) search.
 */
struct NegamaxConfig {
    int depth = 4;           ///< Lookahead depth (number of PUT decisions, both players)
    int candidate_n = 22;    ///< Number of candidate actions (22 = all possible placement options)
    VsBeamConfig vs_config;          ///< Root node: high-quality VS beam search config
    VsBeamConfig interior_vs_config; ///< Interior nodes: lightweight config (small beam, no attack search)
    bool use_interior_config = false; ///< If false, vs_config is used at all depths
};

/**
 * @struct NegamaxResult
 * @brief Output returned by negamaxSearch.
 */
struct NegamaxResult {
    int best_action = 0;
    float best_eval = -100000.0f;
    std::vector<std::pair<int, float>> candidate_evals;
};

/**
 * @brief Evaluates a match state at a leaf node from the perspective of my_id.
 * Returns: (my_score + my_potential) - (opp_score + opp_potential).
 */
float evaluateMatchState(const PuyotanMatch& match, int my_id, const VsBeamEvalWeights& weights) noexcept;

/**
 * @brief Performs a Negamax (minimax) search over PuyotanMatch states up to the specified decision depth.
 *
 * Uses vsBeamSearchTopN at each step to narrow candidates, and steps the deterministic PuyotanMatch
 * until the next decision point.
 */
NegamaxResult negamaxSearch(const PuyotanMatch& match, int my_id, const NegamaxConfig& cfg) noexcept;

} // namespace puyotan::search
