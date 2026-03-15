#pragma once

#include <puyotan/common/config.hpp>
#include <puyotan/core/chain.hpp>
#include <algorithm>

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
    static int calculateStepScore(const ErasureData& data, int chain_number) {
        if (!data.erased || data.num_erased == 0) {
            return 0;
        }

        int chain_bonus = getChainBonus(chain_number);
        int color_bonus = getColorBonus(data.num_colors);
        int group_bonus = 0;
        for (int g = 0; g < data.num_groups; ++g) {
            group_bonus += getGroupBonus(data.group_sizes[g]);
        }

        int total_bonus = chain_bonus + color_bonus + group_bonus;
        if (total_bonus < 1) {
            total_bonus = 1;
        }

        return (data.num_erased * 10) * total_bonus;
    }

private:
    static constexpr int getChainBonus(int chain) {
        int idx = (chain < 1) ? 0 : (chain - 1);
        if (idx >= config::Score::kChainBonusesSize) {
            return config::Score::kChainBonuses[config::Score::kChainBonusesSize - 1];
        }
        return config::Score::kChainBonuses[idx];
    }

    static constexpr int getColorBonus(int count) {
        if (count < 1) return 0;
        if (count >= config::Score::kColorBonusesSize) {
            return config::Score::kColorBonuses[config::Score::kColorBonusesSize - 1];
        }
        return config::Score::kColorBonuses[count];
    }

    static constexpr int getGroupBonus(int size) {
        int idx = size - config::Rule::kConnectCount;
        if (idx < 0) return 0;
        if (idx >= config::Score::kGroupBonusesSize) {
            return config::Score::kGroupBonuses[config::Score::kGroupBonusesSize - 1];
        }
        return config::Score::kGroupBonuses[idx];
    }
};

} // namespace puyotan
