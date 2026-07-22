#include <puyotan/search/attack_finder.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>
#include <algorithm>
#include <queue>

namespace puyotan::search {

namespace {

struct SearchState {
    Board field;
    int depth;
    int first_action;
    int accum_turns;
};

struct AttackAction {
    int idx;
    int ax;
    int sx;
    int axis_dy;
    int sub_dy;
};

const std::vector<AttackAction>& getPutActions() noexcept {
    static const auto v = []() {
        std::vector<AttackAction> r;
        for (int i = 0; i < kNumRLActions; ++i) {
            Action a = getRLAction(i);
            if (a.type == ActionType::Put) {
                const int rot = static_cast<int>(a.rotation) & 3;
                const int ax = a.x;
                const int sx = ax + kSubDx[rot];
                const int axis_dy = kAxisDy[rot];
                const int sub_dy = kSubDySimple[rot];
                r.emplace_back(i, ax, sx, axis_dy, sub_dy);
            }
        }
        return r;
    }();
    return v;
}

const std::vector<AttackAction>& getZoroActions() noexcept {
    static const auto v = []() {
        std::vector<AttackAction> r;
        for (int i = 0; i < kNumRLActions; ++i) {
            Action a = getRLAction(i);
            if (a.type != ActionType::Put)
                continue;
            if (a.rotation == Rotation::Down || a.rotation == Rotation::Left)
                continue;

            const int rot = static_cast<int>(a.rotation) & 3;
            const int ax = a.x;
            const int sx = ax + kSubDx[rot];
            const int axis_dy = kAxisDy[rot];
            const int sub_dy = kSubDySimple[rot];
            r.emplace_back(i, ax, sx, axis_dy, sub_dy);
        }
        return r;
    }();
    return v;
}

__forceinline uint32_t packHeights(const Board& field) noexcept {
    uint32_t packed = 0;
    for (int col = 0; col < 6; ++col) {
        packed |= (static_cast<uint32_t>(field.getColumnHeight(col)) << (col << 2));
    }
    return packed;
}

} // anonymous namespace

std::vector<AttackCandidate> collectAttackCandidates(
    const Board& field,
    const Tsumo& tsumo,
    int tsumo_base,
    int max_depth
) noexcept {
    // Safety cap: Exhaustive search tree grows by ~22x per depth level.
    // Cap max_depth to 4 (at most ~230k nodes) to guarantee instant execution and zero OOM risk.
    max_depth = std::min(max_depth, 4);

    std::vector<AttackCandidate> candidates;
    candidates.reserve(64);

    std::vector<SearchState> current_layer;
    current_layer.reserve(128);
    current_layer.push_back({field, 0, -1, 0});

    for (int d = 0; d < max_depth; ++d) {
        PuyoPiece piece = tsumo.get(tsumo_base + d);
        const bool is_zoro = (piece.axis == piece.sub);
        const auto& actions = is_zoro ? getZoroActions() : getPutActions();

        std::vector<SearchState> next_layer;
        next_layer.reserve(current_layer.size() * actions.size());

        for (const auto& state : current_layer) {
            uint32_t packed_heights = packHeights(state.field);

            for (const auto& act : actions) {
                const int ax = act.ax;
                const int sx = act.sx;

                const int h_axis = (packed_heights >> (ax << 2)) & 0xFu;
                const int h_sub  = (packed_heights >> (sx << 2)) & 0xFu;

                const int y_axis = h_axis + act.axis_dy;
                const int y_sub = h_sub + act.sub_dy;

                if (y_axis >= config::Board::kTotalRows || y_sub >= config::Board::kTotalRows)
                    continue;

                Board child = state.field;
                child.dropPiecePairFast(ax, sx, y_axis, y_sub, piece.axis, piece.sub);

                int first_act = (d == 0) ? act.idx : state.first_action;

                ErasureData ed;
                Chain::scanGroups(child, ed, piece.dirty_flag);

                if (ed.num_erased > 0) {
                    int chain_count = 0;
                    int chain_score = 0;

                    while (ed.num_erased > 0) {
                        ++chain_count;
                        chain_score += Scorer::calculateStepScore(ed, chain_count);
                        Chain::applyErasure(child, ed);
                        uint32_t fallen = Gravity::execute(child);
                        Chain::scanGroups(child, ed, fallen);
                    }

                    if (!child.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
                        bool all_clear = child.getOccupied().empty();
                        int prep_turns = d + 1;
                        int total_frames = prep_turns + chain_count;
                        candidates.push_back({
                            first_act,
                            chain_score,
                            chain_count,
                            prep_turns,
                            total_frames,
                            all_clear
                        });
                    }
                } else {
                    if (!child.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
                        next_layer.push_back({child, d + 1, first_act, d + 1});
                    }
                }
            }
        }
        current_layer = std::move(next_layer);
        if (current_layer.empty()) break;
    }

    std::sort(candidates.begin(), candidates.end(), [](const AttackCandidate& a, const AttackCandidate& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.total_frames < b.total_frames;
    });

    return candidates;
}

} // namespace puyotan::search
