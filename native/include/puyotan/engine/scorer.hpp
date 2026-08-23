#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <puyotan/common/config.hpp>
#include <puyotan/core/chain.hpp>

namespace puyotan {

class Scorer {
  public:
    static __forceinline int calculateStepScore(const ErasureData& data,
                                  int chain_number) noexcept {
        const int chain_bonus = getChainBonus(chain_number);
        const int color_bonus = getColorBonus(data.num_colors);
        const int bonus_sum = chain_bonus + color_bonus + data.group_bonus;
        // ★ CMOV 命令 (2命令) による最速の最小値 1 クランプ
        const int total_bonus = std::max(1, bonus_sum);

        return (data.num_erased * 10) * total_bonus;
    }

  private:
    // ★ 型を uint16_t / uint8_t に適正化し、テーブル全体を 78 バイト (1キャッシュライン) に圧縮！
    static constexpr uint16_t kChainBonuses[20] = {
        0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 512
    };

    static constexpr uint8_t kColorBonuses[6] = {
        0, 0, 3, 6, 12, 24
    };

    // わずか 32 バイトの超軽量グループボーナステーブル
    static constexpr auto kPaddedGroupBonuses = []() consteval {
        std::array<uint8_t, 32> arr{};
        for (int i = 0; i < 32; ++i) {
            if (i < config::Rule::kConnectCount) {
                arr[i] = 0;
            } else {
                const int idx = i - config::Rule::kConnectCount;
                arr[i] = static_cast<uint8_t>(config::Score::kGroupBonuses[std::min(
                    idx, config::Score::kGroupBonusesSize - 1)]);
            }
        }
        return arr;
    }();

    static constexpr int getChainBonus(int chain) noexcept {
        assert(chain >= 1);
        const int idx = std::min(chain, 20) - 1;
        return kChainBonuses[idx];
    }

    static constexpr int getColorBonus(int count) noexcept {
        assert(count >= 0 && count < 6);
        return kColorBonuses[count];
    }

    static constexpr int getGroupBonus(int size) noexcept {
        const int idx = std::min(size, 31);
        return kPaddedGroupBonuses[idx];
    }
};

} // namespace puyotan