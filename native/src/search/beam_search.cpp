#include <algorithm>
#include <vector>

#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/env/vector_match.hpp>  // for kNumRLActions, getRLAction
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {
namespace {

// ---------------------------------------------------------------------------
// BeamNode: one candidate board state in the beam
// ---------------------------------------------------------------------------
struct BeamNode {
    Board field;
    float score;
    int   first_action; // RL action index chosen at depth 0
};

// ---------------------------------------------------------------------------
// Simulate placing one tsumo piece (axis + sub) onto the board.
// Performs an instant-drop of both puyos, then resolves the resulting chain.
// Returns the total chain count and score achieved.
// ---------------------------------------------------------------------------
struct PlaceResult {
    Board field;
    int   chain;
    int   score;
    bool  dead;   // true if the placement would overflow the death row
};

PlaceResult simulatePlacement(const Board& src,
                              PuyoPiece    piece,
                              Action       action) noexcept {
    PlaceResult res{src, 0, 0, false};

    const int r = static_cast<int>(action.rotation) & 3;
    const int ax = action.x;
    const int sx = std::clamp(static_cast<int>(ax + kSubDx[r]), 0, config::Board::kWidth - 1);

    const int h_axis = res.field.getColumnHeight(ax);
    const int h_sub  = res.field.getColumnHeight(sx);

    const int y_axis = h_axis + kAxisDy[r];
    const int y_sub  = h_sub + kSubDySimple[r];

    // Check bounds
    if (y_axis >= config::Board::kHeight || y_sub >= config::Board::kHeight) {
        res.dead = true;
        return res;
    }

    res.field.dropNewPiece(ax, y_axis, piece.axis);
    res.field.dropNewPiece(sx, y_sub, piece.sub);

    // Death check: column 2 (kDeathCol), row >= kDeathRow
    if (res.field.getColumnHeight(config::Rule::kDeathCol) > config::Rule::kDeathRow) {
        res.dead = true;
        return res;
    }

    // Resolve chain
    uint32_t color_mask = Chain::kAllColorsMask;
    ErasureData ed = Chain::findGroups(res.field, color_mask);
    while (ed.num_erased > 0) {
        ++res.chain;
        res.score += Scorer::calculateStepScore(ed, res.chain);
        Chain::applyErasure(res.field, ed);
        uint32_t fallen = Gravity::execute(res.field);
        ed = Chain::findGroups(res.field, fallen);
    }

    return res;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// beamSearch
// ---------------------------------------------------------------------------
int beamSearch(const PuyotanPlayer& player,
               const Tsumo&         tsumo_const,
               const BeamConfig&    cfg) noexcept {
    // Tsumo::get() is non-const (lazy generation) so we work with a local copy.
    Tsumo tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    // Initialise beam with a single root node (no action taken yet)
    std::vector<BeamNode> current_beam;
    current_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    std::vector<BeamNode> next_beam;
    next_beam.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    // Seed the beam with the current board state
    current_beam.push_back({player.field, 0.0f, -1});

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        PuyoPiece piece = tsumo.get(tsumo_base + depth);
        next_beam.clear();

        for (const BeamNode& node : current_beam) {
            for (int ai = 0; ai < kNumRLActions; ++ai) {
                Action act = getRLAction(ai);
                if (act.type != ActionType::Put) continue;

                PlaceResult pr = simulatePlacement(node.field, piece, act);
                if (pr.dead) continue;

                float eval = BeamEvaluator::evaluate(
                    pr.field, cfg.eval_weights, pr.chain, pr.score);

                int first = (depth == 0) ? ai : node.first_action;
                next_beam.push_back({pr.field, eval, first});
            }
        }

        if (next_beam.empty()) break;

        // Sort descending by score and trim to beam_width
        int keep = std::min(static_cast<int>(next_beam.size()), cfg.beam_width);
        std::partial_sort(
            next_beam.begin(),
            next_beam.begin() + keep,
            next_beam.end(),
            [](const BeamNode& a, const BeamNode& b) { return a.score > b.score; });
        next_beam.resize(keep);

        std::swap(current_beam, next_beam);
    }

    // Return the action from the best surviving leaf
    if (!current_beam.empty() && current_beam[0].first_action >= 0)
        return current_beam[0].first_action;

    // Fallback: return action 0 (Up, col 0) if search found nothing valid
    return 0;
}

} // namespace puyotan::search
