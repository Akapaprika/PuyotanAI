#pragma once

#include <puyotan/common/config.hpp>
#include <puyotan/core/chain.hpp>
#include <algorithm>
#include <cassert>

namespace puyotan {

/**
 * @class Scorer
 * @brief Implements the standard Puyo Puyo scoring algorithm.
 * 
 * Score = (Num Erased * 10) * (Chain Bonus + Color Bonus + Group Bonus)
 */
class Scorer {
public:
    /**
     * @brief Calculates the score for a single chain step.
     * @param data Erasure information (number of colors, groups, etc.).
     * @param chain_number Current chain index (1-indexed).
     * @return Total score points for this step.
     */
    static int calculateStepScore(const ErasureData& data, int chain_number) noexcept {
        const int chain_bonus = getChainBonus(chain_number);
        const int color_bonus = getColorBonus(data.num_colors);
        int group_bonus = 0;
        for (int g = 0; g < data.num_groups; ++g) {
            group_bonus += getGroupBonus(data.group_sizes[g]);
        }

        const int bonus_sum = chain_bonus + color_bonus + group_bonus;
        // Branchless CMOV-eliminated max(1, sum): sum is never negative, so if sum == 0 we add 1.
        const int total_bonus = bonus_sum + (bonus_sum == 0);
        
        // (data.num_erased * 10) is computed in parallel with bonus_sum on the CPU 
        // because it has no dependencies, minimizing critical path latency.
        return (data.num_erased * 10) * total_bonus;
    }

    static_assert(config::Score::kChainBonusesSize >= 19, "Chain bonus array should cover standard max chains");
    static_assert(config::Score::kColorBonusesSize >= 5, "Color bonus array must cover all 5 colors");
    static_assert(config::Score::kGroupBonusesSize >= 1, "Group bonus array cannot be empty");

private:
    static constexpr int getChainBonus(int chain) noexcept {
        // Chain count is bounded by board dimensions (max 19-21); assert is sufficient.
        assert(chain >= 1 && chain <= config::Score::kChainBonusesSize);
        return config::Score::kChainBonuses[chain - 1];
    }

    static constexpr int getColorBonus(int count) noexcept {
        // Number of colors is fixed at 4 (+1 ojama); assert is sufficient.
        assert(count >= 1 && count < config::Score::kColorBonusesSize);
        return config::Score::kColorBonuses[count];
    }

    // Generates a fully padded group bonus array to avoid std::min(idx, max_idx) bounding entirely.
    // Max board size is 6x14=84 cells, so size-4 index max is ~80. Array size 128 is safe.
    static constexpr auto kPaddedGroupBonuses = []() consteval {
        std::array<int, 128> arr{};
        for (int i = 0; i < 128; ++i) {
            if (i < config::Score::kGroupBonusesSize) {
                arr[i] = config::Score::kGroupBonuses[i];
            } else {
                arr[i] = config::Score::kGroupBonuses[config::Score::kGroupBonusesSize - 1]; // pad max
            }
        }
        return arr;
    }();

    static constexpr int getGroupBonus(int size) noexcept {
        const int idx = size - config::Rule::kConnectCount;
        assert(idx >= 0 && idx < 128); // Safe bounding limit
        // O(1) direct lookup without std::min clamping (CMOV eliminated)
        return kPaddedGroupBonuses[idx];
    }
};

} // namespace puyotan
