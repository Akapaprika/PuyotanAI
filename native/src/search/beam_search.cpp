#include <algorithm>
#include <omp.h>
#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/attack_finder.hpp>
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

struct BeamAction {
    int idx;
    int ax;
    int sx;
    int axis_dy;
    int sub_dy;
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
                r.emplace_back(i, ax, sx, axis_dy, sub_dy);
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
    for (int col = 0; col < 6; ++col) {
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
    // Pre-calculate effective fire bias for VS mode if attack search is enabled
    // -----------------------------------------------------------------------
    float effective_bias = 1.0f;
    if constexpr (HasFireBias) {
        effective_bias = cfg.eval_weights.fire_bias;

        if (cfg.enable_attack_search) {
            const auto& aw = cfg.eval_weights;
            const int total_incoming = cfg.context.my_active_ojama + cfg.context.my_non_active_ojama;
            if (total_incoming > 0) {
                effective_bias *= aw.incoming_threat_bias;
            }

            auto my_attacks = collectAttackCandidates(player.field, tsumo, player.active_next_pos, std::min(cfg.look_ahead, 3));
            auto enemy_attacks = collectAttackCandidates(cfg.context.enemy_field, tsumo, cfg.context.enemy_active_next_pos, std::min(cfg.look_ahead, 3));

            if (!my_attacks.empty()) {
                const auto& best_attack = my_attacks[0];
                int attack_ojama = best_attack.score / config::Score::kTargetScore;

                // 1. 【対応（カウンター相殺）判定】
                // 相手からの予告おじゃまを相殺できる十分な攻撃がある場合
                if (total_incoming > 0 &&
                    attack_ojama >= static_cast<int>(total_incoming / aw.counter_ratio)) {
                    effective_bias *= aw.counter_attack_bias;
                }
                // 相手が無防備で致死攻撃を送れる場合
                else if (cfg.context.enemy_action_type != ActionType::Chain &&
                         cfg.context.enemy_action_type != ActionType::ChainFall &&
                         attack_ojama >= aw.lethal_ojama_threshold) {
                    effective_bias *= aw.lethal_attack_bias;
                }

                // 2. 【催促（ハラス）判定】
                // 最初のおじゃまが相手に届くステップ数 (発火ツモ数 + 1ステップ) < 相手が発火可能になるツモ数
                // または 相手より早く攻撃を発火できる場合
                if (!enemy_attacks.empty()) {
                    const int my_first_ojama_step = best_attack.prepare_turns + 1;
                    const int enemy_start_step = enemy_attacks[0].prepare_turns;

                    if (enemy_start_step >= my_first_ojama_step || enemy_start_step > best_attack.prepare_turns) {
                        effective_bias *= aw.timing_advantage_bias;
                    }
                } else {
                    // 相手に即時発火可能な攻撃がない場合（完全無防備・催促チャンス）
                    effective_bias *= aw.timing_advantage_bias;
                }
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

                float eval = 0.0f;
                float total_score = 0.0f;
                float next_accum = node.accum_score;

                if constexpr (HasFireBias) {
                    eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights, &cfg.context);
                    // Branchless computation: blend scale and score_add to avoid CPU branch mispredictions
                    const bool is_fire = (pr.score > 0);
                    const float scale = is_fire ? 1.0f : cfg.eval_weights.potential_score_scale;
                    const float score_add = static_cast<float>(pr.score) * effective_bias;
                    next_accum = node.accum_score * scale + score_add;
                    total_score = next_accum + eval;
                } else {
                    eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights);
                    next_accum += static_cast<float>(pr.score);
                    total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;
                }

                int first = (depth == 0) ? entry.idx : node.first_action;
                tl_next_beam.emplace_back(std::move(pr.field), total_score, next_accum, first);
            }
        }

        if (tl_next_beam.empty())
            break;

        // Sort descending by score and trim to beam_width using lightweight index sort
        int keep = std::min(static_cast<int>(tl_next_beam.size()), cfg.beam_width);
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

    // Return the action and its expected score from the best surviving leaf
    if (!tl_current_beam.empty() && tl_current_beam[0].first_action >= 0)
        return {tl_current_beam[0].first_action, tl_current_beam[0].score};

    // Fallback: return action 0 (Up, col 0) if search found nothing valid
    return {0, -10000.0f};
}

std::pair<int, float> soloBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo& tsumo_const,
                                     const SoloBeamConfig& cfg) noexcept {
    return beamSearchImpl<SoloBeamConfig, SoloBeamEvaluator, false>(player, tsumo_const, cfg);
}

std::pair<int, float> vsBeamSearch(const PuyotanPlayer& player,
                                   const Tsumo& tsumo_const,
                                   const VsBeamConfig& cfg) noexcept {
    return beamSearchImpl<VsBeamConfig, VsBeamEvaluator, true>(player, tsumo_const, cfg);
}

} // namespace puyotan::search
