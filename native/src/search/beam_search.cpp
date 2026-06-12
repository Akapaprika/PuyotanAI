#include <algorithm>
#include <vector>
#include <omp.h>

#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {
namespace {

// ---------------------------------------------------------------------------
// BeamNode: one candidate board state in the beam
// ---------------------------------------------------------------------------
struct BeamNode {
    Board field;
    float score;
    float accum_score;
    int   first_action; // RL action index chosen at depth 0
};

struct BeamAction {
    int idx;
    int ax;
    int sx;
    int axis_dy;
    int sub_dy;
    bool is_death_col_related;
};

// Returns all Put actions precomputed
const std::vector<BeamAction>& getPutActions() noexcept {
    static const auto v = []() {
        std::vector<BeamAction> r;
        for (int i = 0; i < kNumRLActions; ++i) {
            Action a = getRLAction(i);
            if (a.type == ActionType::Put) {
                const int rot = static_cast<int>(a.rotation) & 3;
                const int ax = a.x;
                const int sx = ax + kSubDx[rot];
                const int axis_dy = kAxisDy[rot];
                const int sub_dy = kSubDySimple[rot];
                const bool is_death_col_related = (ax == config::Rule::kDeathCol || sx == config::Rule::kDeathCol);
                r.push_back({i, ax, sx, axis_dy, sub_dy, is_death_col_related});
            }
        }
        return r;
    }();
    return v;
}

// Returns Zoro actions precomputed
const std::vector<BeamAction>& getZoroActions() noexcept {
    static const auto v = []() {
        std::vector<BeamAction> r;
        for (int i = 0; i < kNumRLActions; ++i) {
            Action a = getRLAction(i);
            if (a.type != ActionType::Put) continue;
            if (a.rotation == Rotation::Down || a.rotation == Rotation::Left) continue;

            const int rot = static_cast<int>(a.rotation) & 3;
            const int ax = a.x;
            const int sx = ax + kSubDx[rot];
            const int axis_dy = kAxisDy[rot];
            const int sub_dy = kSubDySimple[rot];
            const bool is_death_col_related = (ax == config::Rule::kDeathCol || sx == config::Rule::kDeathCol);
            r.push_back({i, ax, sx, axis_dy, sub_dy, is_death_col_related});
        }
        return r;
    }();
    return v;
}

struct ScoreIdx {
    float score;
    int   idx;
};

// Thread-local vector to avoid dynamic allocation overhead in the hot loop
thread_local std::vector<ScoreIdx> tl_sort_buf;

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
                              const BeamAction& action) noexcept {
    const int ax = action.ax;
    const int sx = action.sx;

    // Read heights from src BEFORE copying — avoids 96-byte Board copy on dead placements.
    const int h_axis = src.getColumnHeight(ax);
    const int h_sub  = src.getColumnHeight(sx);

    const int y_axis = h_axis + action.axis_dy;
    const int y_sub  = h_sub  + action.sub_dy;

    // Early-out: bounds check before the expensive Board copy.
    // Changed boundary check to kTotalRows to allow placements above visible height (13) but below total bitmap rows (15).
    if (y_axis >= config::Board::kTotalRows || y_sub >= config::Board::kTotalRows) [[unlikely]] {
        return {Board{}, 0, 0, true};
    }

    PlaceResult res{src, 0, 0, false}; // 96-byte copy only for valid placements
    res.field.dropNewPiece(ax, y_axis, piece.axis);
    res.field.dropNewPiece(sx, y_sub, piece.sub);

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

    // Death check (deferred until after chains resolve):
    // Check if the death cell is occupied after the chain finishes.
    if (res.field.get(config::Rule::kDeathCol, config::Rule::kDeathRow) != Cell::Empty) [[unlikely]] {
        res.dead = true;
        return res;
    }

    return res;
}

struct TsumoPattern {
    int c1;
    int c2;
    float weight;
    bool is_zoro;
};

// 10 unique combinations of tsumo colors (4 colors)
const TsumoPattern kTsumoPatterns[10] = {
    {0, 0, 1.0f / 16.0f, true},
    {0, 1, 2.0f / 16.0f, false},
    {0, 2, 2.0f / 16.0f, false},
    {0, 3, 2.0f / 16.0f, false},
    {1, 1, 1.0f / 16.0f, true},
    {1, 2, 2.0f / 16.0f, false},
    {1, 3, 2.0f / 16.0f, false},
    {2, 2, 1.0f / 16.0f, true},
    {2, 3, 2.0f / 16.0f, false},
    {3, 3, 1.0f / 16.0f, true}
};

// Evaluate single tsumo for Expectimax (at deep levels)
template<bool HasOjama>
inline float evaluateSingleTsumo(const Board& field, const TsumoPattern& pat, const BeamConfig& cfg) noexcept {
    float max_eval = -1e9f;
    Cell axis = static_cast<Cell>(pat.c1);
    Cell sub = static_cast<Cell>(pat.c2);
    uint8_t dirty = (1u << pat.c1) | (1u << pat.c2);
    PuyoPiece piece{axis, sub, dirty, 0};

    const auto& actions = pat.is_zoro ? getZoroActions() : getPutActions();
    for (const auto& entry : actions) {
        PlaceResult pr = simulatePlacement(field, piece, entry);
        if (pr.dead) continue;

        // Add the immediate chain score of the placement to the evaluation
        float eval = BeamEvaluator::evaluate<false, false, HasOjama>(pr.field, cfg.eval_weights) +
                     static_cast<float>(pr.score) * cfg.eval_weights.potential_score_scale;
        max_eval = (eval > max_eval) ? eval : max_eval;
    }
    if (max_eval < -1e8f) {
        max_eval = -10000.0f;
    }
    return max_eval;
}

// Evaluate two steps of invisible tsumo (4th and 5th steps)
template<bool HasOjama>
inline float evaluateTwoSteps(const Board& field, const BeamConfig& cfg) noexcept {
    float expected_score_4 = 0.0f;

    for (const auto& pat4 : kTsumoPatterns) {
        Cell axis4 = static_cast<Cell>(pat4.c1);
        Cell sub4 = static_cast<Cell>(pat4.c2);
        uint8_t dirty4 = (1u << pat4.c1) | (1u << pat4.c2);
        PuyoPiece piece4{axis4, sub4, dirty4, 0};

        float max_eval_4 = -1e9f;

        const auto& actions4 = pat4.is_zoro ? getZoroActions() : getPutActions();
        for (const auto& entry4 : actions4) {
            PlaceResult pr4 = simulatePlacement(field, piece4, entry4);
            if (pr4.dead) continue;

            float expected_score_5 = 0.0f;
            for (const auto& pat5 : kTsumoPatterns) {
                float eval_5 = evaluateSingleTsumo<HasOjama>(pr4.field, pat5, cfg);
                expected_score_5 += eval_5 * pat5.weight;
            }

            // Include the immediate chain score of the 4th step placement in the evaluation
            float eval_4 = expected_score_5 + static_cast<float>(pr4.score) * cfg.eval_weights.potential_score_scale;
            max_eval_4 = (eval_4 > max_eval_4) ? eval_4 : max_eval_4;
        }

        if (max_eval_4 < -1e8f) {
            max_eval_4 = -10000.0f;
        }
        expected_score_4 += max_eval_4 * pat4.weight;
    }

    return expected_score_4;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// beamSearch
// ---------------------------------------------------------------------------
template<bool UseFastPotential, bool HasOjama>
std::pair<int, float> beamSearchImpl(const PuyotanPlayer& player,
                                     const Tsumo&         tsumo_const,
                                     const BeamConfig&    cfg) noexcept {
    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    // -----------------------------------------------------------------------
    // Fire scan: try all actions with the current tsumo and find the best
    // immediate chain.  This is compared against the beam result at the end
    // to decide whether firing now is better than continuing to build.
    // -----------------------------------------------------------------------
    int   fire_best_action = -1;
    float fire_best_score  = 0.0f;
    {
        PuyoPiece piece0 = tsumo.get(tsumo_base + 0);
        const bool is_zoro0 = (piece0.axis == piece0.sub);
        const auto& actions0 = is_zoro0 ? getZoroActions() : getPutActions();
        for (const auto& entry : actions0) {
            PlaceResult pr = simulatePlacement(player.field, piece0, entry);
            if (pr.dead || pr.score == 0) continue;
            float s = static_cast<float>(pr.score);
            if (s > fire_best_score) {
                fire_best_score  = s;
                fire_best_action = entry.idx;
            }
        }
    }

    // Initialise beam with a single root node (no action taken yet)
    std::vector<BeamNode> current_beam;
    current_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    std::vector<BeamNode> next_beam;
    next_beam.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    // Seed the beam with the current board state
    current_beam.push_back({player.field, 0.0f, 0.0f, -1});

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        if (depth == 3) {
            // Invisible steps starting from 4th (depth=3)
            const int num_nodes = static_cast<int>(current_beam.size());

            if (cfg.look_ahead == 4) {
                // 1-step Expectimax (4th step only)
                #pragma omp parallel for num_threads(2) schedule(static)
                for (int i = 0; i < num_nodes; ++i) {
                    BeamNode& node = current_beam[i];
                    float expected_score = 0.0f;
                    for (const auto& pat : kTsumoPatterns) {
                        float eval = evaluateSingleTsumo<HasOjama>(node.field, pat, cfg);
                        expected_score += eval * pat.weight;
                    }
                    node.score = node.accum_score * cfg.eval_weights.potential_score_scale + expected_score;
                }
            } else {
                // 2-step Expectimax (4th & 5th steps)
                #pragma omp parallel for num_threads(2) schedule(static)
                for (int i = 0; i < num_nodes; ++i) {
                    BeamNode& node = current_beam[i];
                    node.score = node.accum_score * cfg.eval_weights.potential_score_scale + evaluateTwoSteps<HasOjama>(node.field, cfg);
                }
            }

            // Sort descending by score
            std::sort(
                current_beam.begin(),
                current_beam.end(),
                [](const BeamNode& a, const BeamNode& b) { return a.score > b.score; });
            break; // We have completed the search
        }

        PuyoPiece piece = tsumo.get(tsumo_base + depth);
        const bool is_zoro = (piece.axis == piece.sub);
        next_beam.clear();

        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        for (const BeamNode& node : current_beam) {
            for (const auto& entry : actions) {
                PlaceResult pr = simulatePlacement(node.field, piece, entry);
                if (pr.dead) continue;

                float eval = BeamEvaluator::evaluate<true, UseFastPotential, HasOjama>(pr.field, cfg.eval_weights);
                float next_accum = node.accum_score + static_cast<float>(pr.score);
                float total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;

                int first = (depth == 0) ? entry.idx : node.first_action;
                next_beam.push_back({pr.field, total_score, next_accum, first});
            }
        }

        if (next_beam.empty()) break;

        // Sort descending by score and trim to beam_width using lightweight index sort
        int keep = std::min(static_cast<int>(next_beam.size()), cfg.beam_width);
        tl_sort_buf.resize(next_beam.size());
        for (std::size_t i = 0; i < next_beam.size(); ++i) {
            tl_sort_buf[i] = {next_beam[i].score, static_cast<int>(i)};
        }

        std::nth_element(
            tl_sort_buf.begin(),
            tl_sort_buf.begin() + keep,
            tl_sort_buf.end(),
            [](const ScoreIdx& a, const ScoreIdx& b) { return a.score > b.score; });

        std::sort(
            tl_sort_buf.begin(),
            tl_sort_buf.begin() + keep,
            [](const ScoreIdx& a, const ScoreIdx& b) { return a.score > b.score; });

        current_beam.resize(keep);
        for (int i = 0; i < keep; ++i) {
            current_beam[i] = std::move(next_beam[tl_sort_buf[i].idx]);
        }
    }

    // -----------------------------------------------------------------------
    // Fire-vs-beam decision:
    // If the best immediate fire score (scaled by fire_bias) exceeds the beam
    // search result, return the firing action instead.
    // fire_bias > 1.0 : favour firing (chain is "saturated")
    // fire_bias == 1.0: fire only when it strictly beats the beam continuation
    // fire_bias < 1.0 : favour building (need clearly dominant fire to switch)
    // -----------------------------------------------------------------------
    if (fire_best_action >= 0 && !current_beam.empty()) {
        const float beam_score = current_beam[0].score;
        if (fire_best_score * cfg.eval_weights.fire_bias > beam_score) {
            return {fire_best_action, fire_best_score};
        }
    }

    // Return the action and its expected score from the best surviving leaf
    if (!current_beam.empty() && current_beam[0].first_action >= 0)
        return {current_beam[0].first_action, current_beam[0].score};

    // Fallback: return action 0 (Up, col 0) if search found nothing valid
    return {0, -10000.0f};
}

std::pair<int, float> beamSearch(const PuyotanPlayer& player,
                                 const Tsumo&         tsumo_const,
                                 const BeamConfig&    cfg) noexcept {
    const bool has_ojama = !player.field.getBitboard(Cell::Ojama).empty();

    if (cfg.eval_weights.use_fast_potential) {
        if (has_ojama) {
            return beamSearchImpl<true, true>(player, tsumo_const, cfg);
        } else {
            return beamSearchImpl<true, false>(player, tsumo_const, cfg);
        }
    } else {
        if (has_ojama) {
            return beamSearchImpl<false, true>(player, tsumo_const, cfg);
        } else {
            return beamSearchImpl<false, false>(player, tsumo_const, cfg);
        }
    }
}

} // namespace puyotan::search
