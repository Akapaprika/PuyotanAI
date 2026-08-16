#pragma once

#include <algorithm>
#include <vector>
#include <puyotan/common/types.hpp>

namespace puyotan::search {

/**
 * @struct BeamAction
 * @brief Precomputed placement descriptor for a single RL action index.
 *
 * Shared across beam_search, attack_finder, and any future searcher that
 * iterates over legal placements.  Storing all geometric values up-front
 * eliminates repeated calls to getRLAction() inside hot loops.
 */
struct BeamAction {
    int idx;      ///< Flat RL action index
    int ax;       ///< Axis-puyo target column (0-based)
    int sx;       ///< Sub-puyo  target column (0-based)
    int axis_dy;  ///< Axis-puyo row offset from the surface (+1 for Down rotation)
    int sub_dy;   ///< Sub-puyo  row offset from the surface (+1 for Up   rotation)
};

/**
 * @brief Column priority order used for edge-to-center ordering of actions.
 *
 * Columns 0 and 5 (edges) are searched first, then 1 and 4, then 2 and 3.
 * This heuristic ordering reduces beam size needed to find good solutions.
 */
inline constexpr int kColPriority[6] = {0, 2, 4, 5, 3, 1};

/**
 * @brief Comparator: orders BeamActions by edge-first column priority.
 *
 * Shared by getPutActions() and getZoroActions() to eliminate duplicated
 * sort logic. The key is packed as (min_priority << 3) | max_priority so
 * that pairs closer to the board edges sort before pairs near the centre.
 */
inline constexpr auto kColPriorityCmp = [](const BeamAction& a, const BeamAction& b) noexcept {
    const int pa = (std::min(kColPriority[a.ax], kColPriority[a.sx]) << 3)
                 |  std::max(kColPriority[a.ax], kColPriority[a.sx]);
    const int pb = (std::min(kColPriority[b.ax], kColPriority[b.sx]) << 3)
                 |  std::max(kColPriority[b.ax], kColPriority[b.sx]);
    return (pa != pb) ? (pa < pb) : (a.idx < b.idx);
};

/**
 * @brief Returns all Put actions (22 total), sorted edge-first by column priority.
 *
 * The result is lazily initialised once and reused across all searchers.
 * Thread-safe: C++11 guarantees that static local initialisers are
 * executed exactly once, even under concurrent access.
 */
[[nodiscard]] inline const std::vector<BeamAction>& getPutActions() noexcept {
    static const auto v = []() {
        std::vector<BeamAction> r;
        r.reserve(kNumRLActions);
        for (int i = 0; i < kNumRLActions; ++i) {
            Action a = getRLAction(i);
            if (a.type != ActionType::Put)
                continue;
            const int rot     = static_cast<int>(a.rotation) & 3;
            const int ax      = a.x;
            const int sx      = ax + kSubDx[rot];
            const int axis_dy = kAxisDy[rot];
            const int sub_dy  = kSubDySimple[rot];
            r.push_back({i, ax, sx, axis_dy, sub_dy});
        }
        std::sort(r.begin(), r.end(), kColPriorityCmp);
        return r;
    }();
    return v;
}

/**
 * @brief Returns the subset of Put actions valid for Zoro (same-color) pieces.
 *
 * Down and Left rotations produce the same board state as Up and Right for
 * symmetric pieces, so they are excluded to halve the search branching factor.
 * Sorted with the same edge-first column priority as getPutActions().
 *
 * Thread-safe: same static-local guarantee as getPutActions().
 */
[[nodiscard]] inline const std::vector<BeamAction>& getZoroActions() noexcept {
    static const auto v = []() {
        std::vector<BeamAction> r;
        r.reserve(kNumRLActions / 2);
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
            r.push_back({i, ax, sx, axis_dy, sub_dy});
        }
        std::sort(r.begin(), r.end(), kColPriorityCmp);
        return r;
    }();
    return v;
}

} // namespace puyotan::search
