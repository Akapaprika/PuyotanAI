#pragma once

#include <vector>
#include <utility>
#include <cstdint>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_evaluator.hpp>

namespace puyotan::search {

/**
 * @struct CategoryBudgets
 * @brief Specifies beam capacity limits for each move category.
 */
struct CategoryBudgets {
    int build  = 10000;
    int crush  = 5000;
    int strike = 3000;
    int evade  = 2000;

    [[nodiscard]] int total() const noexcept {
        return build + crush + strike + evade;
    }
};

/**
 * @enum class AbsCategory
 * @brief Move classification categories for category-preserving beam pruning.
 */
enum class AbsCategory : uint8_t {
    Build  = 0,
    Crush  = 1,
    Strike = 2,
    Evade  = 3
};

/**
 * @struct AbsConfig
 * @brief Parameters controlling Adversarial Beam Search.
 */
struct AbsConfig {
    int depth = 10;
    bool chain_cutoff_enabled = true;
    CategoryBudgets my_budgets;
    CategoryBudgets opp_budgets;
    VsBeamEvalWeights eval_weights;
};

/**
 * @struct AbsResult
 * @brief Results returned by absSearch.
 */
struct AbsResult {
    int best_action = 0;
    float best_eval = 0.0f;
    std::vector<std::pair<int, float>> candidate_evals;
};

/**
 * @brief Runs an Adversarial Beam Search starting from the given PuyotanMatch.
 * @param match Initial match state.
 * @param my_id Player ID for whom we are searching (0 or 1).
 * @param cfg Config containing depth, budgets, and weights.
 * @return AbsResult containing the best RL action index and evaluation scores.
 */
AbsResult absSearch(const PuyotanMatch& match, int my_id, const AbsConfig& cfg) noexcept;

} // namespace puyotan::search
