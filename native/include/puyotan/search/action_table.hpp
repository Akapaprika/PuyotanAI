#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <puyotan/common/types.hpp>

namespace puyotan::search {

/**
 * @struct BeamAction
 * @brief 最小フットプリント (5 bytes / alignas(4)) でパックした配置記述子
 */
struct alignas(4) BeamAction {
    uint8_t idx;      ///< Flat RL action index (0..21)
    uint8_t ax;       ///< Axis-puyo target column (0..5)
    uint8_t sx;       ///< Sub-puyo  target column (0..5)
    uint8_t axis_dy;  ///< Axis-puyo row offset (0..1)
    uint8_t sub_dy;   ///< Sub-puyo  row offset (0..1)
};

inline constexpr int kColPriority[6] = {0, 2, 4, 5, 3, 1};

inline constexpr auto kColPriorityCmp = [](const BeamAction& a, const BeamAction& b) noexcept {
    const int pa = (std::min(kColPriority[a.ax], kColPriority[a.sx]) << 3)
                 |  std::max(kColPriority[a.ax], kColPriority[a.sx]);
    const int pb = (std::min(kColPriority[b.ax], kColPriority[b.sx]) << 3)
                 |  std::max(kColPriority[b.ax], kColPriority[b.sx]);
    return (pa != pb) ? (pa < pb) : (a.idx < b.idx);
};

namespace detail {

[[nodiscard]] inline std::array<BeamAction, 22> initPutActions() noexcept {
    std::array<BeamAction, 22> r{};
    int count = 0;
    for (int i = 0; i < kNumRLActions; ++i) {
        Action a = getRLAction(i);
        if (a.type != ActionType::Put)
            continue;
        const int rot     = static_cast<int>(a.rotation) & 3;
        const int ax      = a.x;
        const int sx      = ax + kSubDx[rot];
        const int axis_dy = kAxisDy[rot];
        const int sub_dy  = kSubDySimple[rot];
        r[count++] = BeamAction{
            static_cast<uint8_t>(i),
            static_cast<uint8_t>(ax),
            static_cast<uint8_t>(sx),
            static_cast<uint8_t>(axis_dy),
            static_cast<uint8_t>(sub_dy)
        };
    }
    std::sort(r.begin(), r.end(), kColPriorityCmp);
    return r;
}

[[nodiscard]] inline std::array<BeamAction, 11> initZoroActions() noexcept {
    std::array<BeamAction, 11> r{};
    int count = 0;
    for (int i = 0; i < kNumRLActions; ++i) {
        Action a = getRLAction(i);
        if (a.type != ActionType::Put)
            continue;
        if (a.rotation == Rotation::Down || a.rotation == Rotation::Left)
            continue;
        const int rot     = static_cast<int>(a.rotation) & 3;
        const int ax      = a.x;
        const int sx      = ax + kSubDx[rot];
        const int axis_dy = kAxisDy[rot];
        const int sub_dy  = kSubDySimple[rot];
        r[count++] = BeamAction{
            static_cast<uint8_t>(i),
            static_cast<uint8_t>(ax),
            static_cast<uint8_t>(sx),
            static_cast<uint8_t>(axis_dy),
            static_cast<uint8_t>(sub_dy)
        };
    }
    std::sort(r.begin(), r.end(), kColPriorityCmp);
    return r;
}

} // namespace detail

/**
 * @brief Returns all Put actions (22 total), sorted edge-first by column priority.
 */
[[nodiscard]] inline std::span<const BeamAction> getPutActions() noexcept {
    static const std::array<BeamAction, 22> actions = detail::initPutActions();
    return actions;
}

/**
 * @brief Returns the subset of Put actions valid for Zoro (same-color) pieces (11 total).
 */
[[nodiscard]] inline std::span<const BeamAction> getZoroActions() noexcept {
    static const std::array<BeamAction, 11> actions = detail::initZoroActions();
    return actions;
}

/**
 * @brief Pack 6 column heights into a single 32-bit register.
 */
[[nodiscard]] __forceinline uint32_t packHeights(const Board& field) noexcept {
    uint32_t packed = 0;
    #pragma unroll
    for (int col = 0; col < config::Board::kWidth; ++col) {
        packed |= (static_cast<uint32_t>(field.getColumnHeight(col)) << (col << 2));
    }
    return packed;
}

} // namespace puyotan::search