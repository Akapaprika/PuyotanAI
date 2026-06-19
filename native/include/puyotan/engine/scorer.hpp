#pragma once

#include <algorithm>
#include <cassert>
#include <puyotan/common/config.hpp>
#include <puyotan/core/chain.hpp>

namespace puyotan {
class Scorer {
  public:
    static int calculateStepScore(const ErasureData& data,
                                  int chain_number) noexcept {
        const int chain_bonus = getChainBonus(chain_number);
        const int color_bonus = getColorBonus(data.num_colors);
        int group_bonus = 0;
        for (int g = 0; g < data.num_groups; ++g) {
            group_bonus += getGroupBonus(data.group_sizes[g]);
        }

        const int bonus_sum = chain_bonus + color_bonus + group_bonus;
        const int total_bonus = bonus_sum + (bonus_sum == 0);

        return (data.num_erased * 10) * total_bonus;
    }

    static_assert(config::Score::kChainBonusesSize >= 19,
                  "Chain bonus array should cover standard max chains");
    static_assert(config::Score::kColorBonusesSize >= 5,
                  "Color bonus array must cover all 5 colors");
    static_assert(config::Score::kGroupBonusesSize >= 1,
                  "Group bonus array cannot be empty");

  private:
    static constexpr int getChainBonus(int chain) noexcept {
        assert(chain >= 1 && chain <= config::Score::kChainBonusesSize);
        return config::Score::kChainBonuses[chain - 1];
    }

    static constexpr int getColorBonus(int count) noexcept {
        assert(count >= 1 && count < config::Score::kColorBonusesSize);
        return config::Score::kColorBonuses[count];
    }

    // C++23/26 compile-time padded array generator.
    // Maps size 0-3 to 0, and size >= 4 directly to its corresponding bonus
    // without requiring any runtime subtraction.
    static constexpr auto kPaddedGroupBonuses = []() consteval {
        std::array<int, 128> arr{};
        for (int i = 0; i < 128; ++i) {
            if (i < config::Rule::kConnectCount) {
                arr[i] = 0;
            } else {
                const int idx = i - config::Rule::kConnectCount;
                arr[i] = config::Score::kGroupBonuses[std::min(
                    idx, config::Score::kGroupBonusesSize - 1)];
            }
        }
        return arr;
    }();

    static constexpr int getGroupBonus(int size) noexcept {
        assert(size >= 0 && size < 128);
        return kPaddedGroupBonuses[size]; // Direct O(1) array lookup with zero
                                          // subtraction overhead
    }
};
} // namespace puyotan