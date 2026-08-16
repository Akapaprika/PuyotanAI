#include <algorithm>
#include <omp.h>
#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_search.hpp>
#include <vector>
#include <unordered_map>

namespace puyotan::search {
namespace {

// ---------------------------------------------------------------------------
// BeamNode: one candidate board state in the beam
// ---------------------------------------------------------------------------
struct BeamNode {
    Board field;
    float score;
    float accum_score;
    int first_action; // RL action index chosen at depth 0
};

// BeamAction, getPutActions(), and getZoroActions() are defined in action_table.hpp.

struct ScoreIdx {
    float score;
    int idx;
};

// Thread-local vector to avoid dynamic allocation overhead in the hot loop
thread_local std::vector<ScoreIdx> tl_sort_buf;
thread_local std::vector<BeamNode> tl_current_beam;
thread_local std::vector<BeamNode> tl_next_beam;

// Pack 6 column heights into a single 32-bit register to minimize memory spills and popcounts.
__forceinline uint32_t packHeights(const Board& field) noexcept {
    uint32_t packed = 0;
    for (int col = 0; col < config::Board::kWidth; ++col) {
        packed |= (static_cast<uint32_t>(field.getColumnHeight(col)) << (col << 2));
    }
    return packed;
}

// ---------------------------------------------------------------------------
// Simulate placing one tsumo piece (axis + sub) onto the board.
// Performs an instant-drop of both puyos, then resolves the resulting chain.
// Returns the total chain count and score achieved.
// ---------------------------------------------------------------------------
struct PlaceResult {
    Board field;
    int chain;
    int score;
    bool dead; // true if the placement would overflow the death row
};

PlaceResult simulatePlacement(const Board& src, PuyoPiece piece,
                              const BeamAction& action,
                              uint32_t packed_heights) noexcept {
    const int ax = action.ax;
    const int sx = action.sx;

    // Decode heights from the packed register
    const int h_axis = (packed_heights >> (ax << 2)) & 0xFu;
    const int h_sub  = (packed_heights >> (sx << 2)) & 0xFu;

    const int y_axis = h_axis + action.axis_dy;
    const int y_sub = h_sub + action.sub_dy;

    // Early-out: bounds check before the expensive Board copy.
    if (y_axis >= config::Board::kTotalRows ||
        y_sub >= config::Board::kTotalRows) [[unlikely]] {
        return {Board{}, 0, 0, true};
    }

    PlaceResult res{src, 0, 0, false}; // 96-byte copy only for valid placements
    res.field.dropPiecePairFast(ax, sx, y_axis, y_sub, piece.axis, piece.sub);

    // Resolve chain
    // 【超強力最適化】
    // 1ステップ目の連鎖判定のみ、新しく置いたぷよの色（最大2色）だけで接続判定を走らせます。
    // これにより、連鎖が起きない約90%のノードにおいて、scanGroups
    // の処理コストが 50%〜75% 削減されます。
    ErasureData ed;
    Chain::scanGroups(res.field, ed, piece.dirty_flag);
    while (ed.num_erased > 0) {
        ++res.chain;
        res.score += Scorer::calculateStepScore(ed, res.chain);
        Chain::applyErasure(res.field, ed);

        // 2ステップ目（連鎖が継続したとき）以降は、おじゃまや他の色ぷよが連動して消える可能性があるため、
        // 4色すべて（kAllColorsMask）をターゲットにして通常通り解決します。
        uint32_t fallen = Gravity::execute(res.field);
        Chain::scanGroups(res.field, ed, fallen);
    }

    // Death check (deferred until after all chains resolve).
    // Must be AFTER chain resolution: a chain can clear puyos from the death
    // cell (col 2, row 11), allowing the player to survive a seemingly fatal
    // placement.
    if (res.field.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow))
        [[unlikely]] {
        res.dead = true;
        return res;
    }

    return res;
}

} // anonymous namespace

template <typename ConfigType, typename EvaluatorType, bool HasFireBias = false>
std::pair<int, float> beamSearchImpl(const PuyotanPlayer& player,
                                     const Tsumo& tsumo_const,
                                     const ConfigType& cfg) noexcept {
    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    // -----------------------------------------------------------------------
    // Fire scan: try all actions with the current tsumo and find the best
    // immediate chain.  This is compared against the beam result at the end
    // to decide whether firing now is better than continuing to build.
    // -----------------------------------------------------------------------
    int fire_best_action = -1;
    float fire_best_score = 0.0f;
    {
        uint32_t packed_heights_root = packHeights(player.field);
        PuyoPiece piece0 = tsumo.get(tsumo_base + 0);
        const bool is_zoro0 = (piece0.axis == piece0.sub);
        const auto& actions0 = is_zoro0 ? getZoroActions() : getPutActions();
        for (const auto& entry : actions0) {
            PlaceResult pr = simulatePlacement(player.field, piece0, entry, packed_heights_root);
            if (pr.dead || pr.score == 0)
                continue;
            float s = static_cast<float>(pr.score);
            if (s > fire_best_score) {
                fire_best_score = s;
                fire_best_action = entry.idx;
            }
        }
    }

    // Initialise beam with a single root node (no action taken yet)
    tl_current_beam.clear();
    tl_current_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    tl_next_beam.clear();
    tl_next_beam.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    // Seed the beam with the current board state
    tl_current_beam.emplace_back(player.field, 0.0f, 0.0f, -1);

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        PuyoPiece piece = tsumo.get(tsumo_base + depth);
        const bool is_zoro = (piece.axis == piece.sub);
        tl_next_beam.clear();

        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        for (const BeamNode& node : tl_current_beam) {
            uint32_t packed_heights = packHeights(node.field);
            for (const auto& entry : actions) {
                PlaceResult pr = simulatePlacement(node.field, piece, entry, packed_heights);
                if (pr.dead)
                    continue;

                float eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights);
                float next_accum =
                    node.accum_score + static_cast<float>(pr.score);
                float total_score =
                    next_accum * cfg.eval_weights.potential_score_scale + eval;

                int first = (depth == 0) ? entry.idx : node.first_action;
                tl_next_beam.emplace_back(pr.field, total_score, next_accum, first);
            }
        }

        if (tl_next_beam.empty())
            break;

        // Calculate dynamic target beam width for current depth (tapered search)
        int target_beam_width = cfg.beam_width;
        if (cfg.min_beam_width_ratio < 1.0f && cfg.look_ahead > 1) {
            if (depth <= cfg.full_beam_depth) {
                target_beam_width = cfg.beam_width;
            } else {
                const float max_decay_steps = static_cast<float>(cfg.look_ahead - 1 - cfg.full_beam_depth);
                if (max_decay_steps > 0.0f) {
                    const float progress = static_cast<float>(depth - cfg.full_beam_depth) / max_decay_steps;
                    const float ratio = 1.0f - (1.0f - cfg.min_beam_width_ratio) * progress;
                    target_beam_width = std::max(1, static_cast<int>(cfg.beam_width * ratio));
                }
            }
        }

        // Sort descending by score and trim to target_beam_width using lightweight index sort
        int keep = std::min(static_cast<int>(tl_next_beam.size()), target_beam_width);
        tl_sort_buf.resize(tl_next_beam.size());
        for (std::size_t i = 0; i < tl_next_beam.size(); ++i) {
            tl_sort_buf[i] = {tl_next_beam[i].score, static_cast<int>(i)};
        }

        if (cfg.dbs_max_similar >= 1) {
            std::sort(tl_sort_buf.begin(), tl_sort_buf.end(),
                      [](const ScoreIdx& a, const ScoreIdx& b) {
                          return a.score > b.score;
                      });

            thread_local std::unordered_map<uint32_t, int> tl_dbs_map;
            tl_dbs_map.clear();

            tl_current_beam.clear();
            for (const auto& item : tl_sort_buf) {
                BeamNode& cand = tl_next_beam[item.idx];
                uint32_t key = packHeights(cand.field);

                int& count = tl_dbs_map[key];
                if (count < cfg.dbs_max_similar) {
                    count++;
                    tl_current_beam.push_back(std::move(cand));
                    if (static_cast<int>(tl_current_beam.size()) == keep) {
                        break;
                    }
                }
            }
        } else {
            std::nth_element(tl_sort_buf.begin(), tl_sort_buf.begin() + keep,
                             tl_sort_buf.end(),
                             [](const ScoreIdx& a, const ScoreIdx& b) {
                                 return a.score > b.score;
                             });

            std::sort(tl_sort_buf.begin(), tl_sort_buf.begin() + keep,
                      [](const ScoreIdx& a, const ScoreIdx& b) {
                          return a.score > b.score;
                      });

            tl_current_beam.resize(keep);
            for (int i = 0; i < keep; ++i) {
                tl_current_beam[i] = std::move(tl_next_beam[tl_sort_buf[i].idx]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fire-vs-beam decision (VS mode only):
    // -----------------------------------------------------------------------
    if constexpr (HasFireBias) {
        if (fire_best_action >= 0 && !tl_current_beam.empty()) {
            const float beam_score = tl_current_beam[0].score;
            if (fire_best_score * cfg.eval_weights.fire_bias > beam_score) {
                return {fire_best_action, fire_best_score};
            }
        }
    }

    // Return the action and its expected score from the best surviving leaf
    if (!tl_current_beam.empty() && tl_current_beam[0].first_action >= 0)
        return {tl_current_beam[0].first_action, tl_current_beam[0].score};

    // Fallback: return action 0 (Up, col 0) if search found nothing valid
    return {0, -10000.0f};
}

std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo& tsumo_const,
                                     const SoloBeamConfig& cfg,
                                     BeamSearchSession* session) noexcept {
    auto res = beamSearchImpl<SoloBeamConfig, SoloBeamEvaluator, false>(player, tsumo_const, cfg);
    if (session) {
        session->update(res.second);
    }
    return res;
}

std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const Tsumo& tsumo_const,
                                   const VsBeamConfig& cfg,
                                   BeamSearchSession* session) noexcept {
    auto res = beamSearchImpl<VsBeamConfig, VsBeamEvaluator, true>(player, tsumo_const, cfg);
    if (session) {
        session->update(res.second);
    }
    return res;
}

std::vector<std::pair<int, float>> vsBeamSearchTopN(const PuyotanPlayer& player,
                                                    const Tsumo& tsumo_const,
                                                    const VsBeamConfig& cfg,
                                                    int top_n) noexcept {
    auto best = vsBeamSearch(player, tsumo_const, cfg);
    std::vector<std::pair<int, float>> res;
    res.push_back(best);
    return res;
}

} // namespace puyotan::search
