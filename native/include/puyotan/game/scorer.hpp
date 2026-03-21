#pragma once

#include <puyotan/common/config.hpp>
#include <puyotan/core/chain.hpp>
#include <algorithm>
#include <cassert>

namespace puyotan {

/**
 * Scorer
 *   Handles calculation of scores based on ErasureData.
 */
class Scorer {
public:
    /**
     * Calculates the score for a single chain step.
     */
    // Called only after Chain::execute confirms num_erased > 0.
    static int calculateStepScore(const ErasureData& data, int chain_number) {
        const int chain_bonus = getChainBonus(chain_number);
        const int color_bonus = getColorBonus(data.num_colors);
        int group_bonus = 0;
        for (int g = 0; g < data.num_groups; ++g) {
            group_bonus += getGroupBonus(data.group_sizes[g]);
        }

        const int total_bonus = std::max(1, chain_bonus + color_bonus + group_bonus);
        return (data.num_erased * 10) * total_bonus;
    }

    static_assert(config::Score::kChainBonusesSize >= 19, "Chain bonus array should cover standard max chains");
    static_assert(config::Score::kColorBonusesSize >= 5, "Color bonus array must cover all 5 colors");
    static_assert(config::Score::kGroupBonusesSize >= 1, "Group bonus array cannot be empty");

private:
    static constexpr int getChainBonus(int chain) {
        // Chain count is bounded by board dimensions (max 19-21); assert is sufficient.
        assert(chain >= 1 && chain <= config::Score::kChainBonusesSize);
        return config::Score::kChainBonuses[chain - 1];
    }

    static constexpr int getColorBonus(int count) {
        // Number of colors is fixed at 4 (+1 ojama); assert is sufficient.
        assert(count >= 1 && count < config::Score::kColorBonusesSize);
        return config::Score::kColorBonuses[count];
    }

    static constexpr int getGroupBonus(int size) {
        const int idx = size - config::Rule::kConnectCount;
        assert(idx >= 0);
        // Groups can physically exceed size 11 (max bonus index); clamp to the last element.
        const int clamped_idx = std::min(idx, config::Score::kGroupBonusesSize - 1);
        return config::Score::kGroupBonuses[clamped_idx];
    }
};

} // namespace puyotan
