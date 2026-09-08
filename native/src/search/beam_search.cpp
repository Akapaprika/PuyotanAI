#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_evaluator.hpp>
#include <puyotan/search/beam_search.hpp>
#include <puyotan/search/depth_dedup_table.hpp>
#include <puyotan/search/transposition_table.hpp>
#include <puyotan/search/zobrist.hpp>

namespace puyotan::search {
namespace {

inline thread_local Board tl_best_leaf_field;

struct alignas(16) BeamNode {
    Board    field;                  // 96 bytes (alignas(16))
    uint64_t hash;                   //  8 bytes
    uint32_t packed_heights_and_act; //  4 bytes: [bit 0..23: packed_heights] [bit 24..31: first_action + 1]
    uint32_t accum_and_flag;         //  4 bytes: [bit 0..30: accum_score]    [bit 31: has_fired_main]

    BeamNode() noexcept = default;

    __forceinline BeamNode(const Board& f, int32_t accum, int first_act,
                           uint32_t packed_h, uint64_t h, bool fired_main) noexcept
        : field(f),
          hash(h),
          packed_heights_and_act((packed_h & 0x00FFFFFFu) | (static_cast<uint32_t>(static_cast<uint8_t>(first_act + 1)) << 24)),
          accum_and_flag((static_cast<uint32_t>(accum) & 0x7FFFFFFFu) | (static_cast<uint32_t>(fired_main) << 31))
    {}

    [[nodiscard]] __forceinline uint32_t packed_heights() const noexcept {
        return packed_heights_and_act & 0x00FFFFFFu;
    }

    [[nodiscard]] __forceinline int first_action() const noexcept {
        return static_cast<int>(packed_heights_and_act >> 24) - 1;
    }

    [[nodiscard]] __forceinline int32_t accum_score() const noexcept {
        return static_cast<int32_t>(accum_and_flag & 0x7FFFFFFFu);
    }

    [[nodiscard]] __forceinline bool has_fired_main() const noexcept {
        return (accum_and_flag >> 31) != 0;
    }
};
static_assert(sizeof(BeamNode) == 112, "BeamNode must be exactly 112 bytes");

// 24バイトの候補記述子
struct CandidateNode {
    uint64_t hash;
    int32_t  score;          
    uint32_t packed_heights; 
    uint32_t parent_idx;     
    uint8_t  action_idx;     
    bool     has_fired_main;
    uint8_t  _pad[2];        
};
static_assert(sizeof(CandidateNode) == 24, "CandidateNode must be exactly 24 bytes");

struct DynamicFlatCountTable {
    struct alignas(8) Entry {
        uint32_t key;
        uint16_t count;
        uint16_t gen;
    };
    static_assert(sizeof(Entry) == 8, "Entry must be exactly 8 bytes");

    std::vector<Entry> table;
    uint32_t mask = 0;
    uint16_t current_gen = 1;

    void ensure_capacity(std::size_t required_capacity) {
        std::size_t cap = 2048;
        while (cap < required_capacity * 2) {
            cap <<= 1;
        }
        if (table.size() != cap) {
            table.assign(cap, Entry{0, 0, 0});
            mask = static_cast<uint32_t>(cap - 1);
            current_gen = 1;
        }
    }

    void clear() noexcept {
        if (++current_gen == 0) [[unlikely]] {
            std::memset(table.data(), 0, table.size() * sizeof(Entry));
            current_gen = 1;
        }
    }

    int get_and_inc(uint32_t key) noexcept {
        std::size_t idx = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) >> 32 & mask;
        while (table[idx].gen == current_gen) {
            if (table[idx].key == key) {
                assert(table[idx].count < 65535);
                return table[idx].count++;
            }
            idx = (idx + 1) & mask;
        }
        table[idx] = {key, 1, current_gen};
        return 0;
    }
};

thread_local std::vector<CandidateNode> tl_candidates;
thread_local std::vector<BeamNode>      tl_current_beam;
thread_local std::vector<BeamNode>      tl_prev_beam;
thread_local DynamicFlatCountTable      tl_dbs_table;

struct PlaceResult {
    Board field;
    int chain;
    int score;
    bool dead;
};

// =========================================================================
// simulatePlacement: 連鎖シミュレーション (ルール完全準拠)
// =========================================================================
__forceinline void simulatePlacement(const Board& src, PuyoPiece piece,
                                     const BeamAction& action,
                                     uint32_t packed_heights,
                                     PlaceResult& out_res) noexcept {
    const int ax = action.ax;
    const int sx = action.sx;

    const int h_axis = (packed_heights >> (ax << 2)) & 0xFu;
    const int h_sub  = (packed_heights >> (sx << 2)) & 0xFu;

    const int y_axis = h_axis + action.axis_dy;
    const int y_sub  = h_sub + action.sub_dy;

    out_res.field = src;
    out_res.chain = 0;
    out_res.score = 0;

    out_res.field.dropPiecePairFast(ax, sx, y_axis, y_sub, piece.axis, piece.sub);

    // 1〜12段目 (消去可能領域) に置かれたぷよがある場合のみスキャン
    if (y_axis < config::Board::kChainableRows || y_sub < config::Board::kChainableRows) {
        ErasureData ed;
        Chain::scanGroups(out_res.field, ed, piece.dirty_flag);

        // 連鎖ループ
        while (ed.num_erased > 0) {
            ++out_res.chain;
            out_res.score += Scorer::calculateStepScore(ed, out_res.chain);
            Chain::applyErasure(out_res.field, ed);

            const uint32_t fallen = Gravity::execute(out_res.field);
            if (fallen == 0) {
                break;
            }

            Chain::scanGroups(out_res.field, ed, fallen);
        }
    }

    // ★ ぷよたんβルール準拠: 連鎖解決後の確定盤面に対して窒息判定を行う
    out_res.dead = out_res.field.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow);
}

// =========================================================================
// N手先読み再帰評価ヘルパー (evaluateMicroPly)
// =========================================================================
// remaining=0 のとき: ポテンシャル評価（基底ケース）
// remaining>0 のとき: 次のツモを展開して再帰呼び出し、最大スコアを返す
int32_t evaluateMicroPly(const Board& field, int32_t accum, uint32_t packed_h, uint64_t hash,
                         const Tsumo& tsumo, int32_t tsumo_idx, int remaining,
                         const SoloBeamConfig& cfg, bool has_fired_main) noexcept {
    if (remaining <= 0 || has_fired_main) {
        if (has_fired_main) {
            return accum * cfg.eval_weights.potential_score_scale;
        }
        int32_t pot = 0;
        if (!tl_tt.get(hash, pot)) {
            pot = computeMaxPotentialScore(field, packed_h);
            tl_tt.put(hash, pot);
        }
        return accum * cfg.eval_weights.potential_score_scale + pot;
    }

    int32_t current_idx = tsumo_idx;
    const PuyoPiece piece = tsumo.get(current_idx);
    const auto& actions = (piece.axis == piece.sub) ? getZoroActions() : getPutActions();
    int32_t best = -1000000000;

    for (const auto& act : actions) {
        PlaceResult pr;
        simulatePlacement(field, piece, act, packed_h, pr);
        if (pr.dead) continue;

        const int32_t step_score = static_cast<int32_t>(pr.score);
        const bool next_fired = has_fired_main ||
                                (cfg.main_chain_threshold > 0 && step_score >= cfg.main_chain_threshold);
        const int32_t next_accum = has_fired_main ? accum : (accum + step_score);
        const uint32_t next_h = packHeights(pr.field);
        const uint64_t next_hash = Zobrist::hashBoard(pr.field);

        const int32_t score = evaluateMicroPly(pr.field, next_accum, next_h, next_hash, tsumo,
                                              current_idx + 1, remaining - 1, cfg, next_fired);
        if (score > best) {
            best = score;
        }
    }
    return best;
}

} // anonymous namespace

template <typename ConfigType, typename EvaluatorType, bool HasFireBias = false>
std::pair<int, int32_t> beamSearchImpl(const PuyotanPlayer& player,
                                       const Tsumo& tsumo_const,
                                       const ConfigType& cfg) noexcept {
    assert(cfg.dbs_max_similar <= 65535 && "dbs_max_similar must not exceed 65535");

    tl_tt.advanceGeneration();
    Zobrist::init();

    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    uint32_t packed_heights_root = packHeights(player.field);
    const uint64_t root_hash = Zobrist::hashBoard(player.field);

    int fire_best_action = -1;
    int32_t fire_best_score = 0;
    if constexpr (HasFireBias) {
        int32_t piece0_idx = tsumo_base;
        PuyoPiece piece0 = tsumo.get(piece0_idx);
        const bool is_zoro0 = (piece0.axis == piece0.sub);
        const auto& actions0 = is_zoro0 ? getZoroActions() : getPutActions();
        
        PlaceResult pr;
        for (const auto& entry : actions0) {
            simulatePlacement(player.field, piece0, entry, packed_heights_root, pr);
            if (pr.dead || pr.score == 0)
                continue;
            int32_t s = static_cast<int32_t>(pr.score);
            if (s > fire_best_score) {
                fire_best_score = s;
                fire_best_action = entry.idx;
            }
        }
    }

    tl_current_beam.clear();
    tl_current_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    tl_prev_beam.clear();
    tl_prev_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    tl_candidates.clear();
    tl_candidates.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    if (cfg.dbs_max_similar >= 1) {
        tl_dbs_table.ensure_capacity(static_cast<std::size_t>(cfg.beam_width));
    }

    tl_current_beam.emplace_back(player.field, 0, -1, packed_heights_root, root_hash, false);

    int best_action = -1;
    int32_t best_score = -1000000000;

    const int occupied_puyos = player.field.getOccupied().popcount();
    const int effective_look_ahead = (cfg.dynamic_lookahead_margin > 0)
        ? std::min(cfg.look_ahead, std::max(1, (cfg.dynamic_lookahead_margin - occupied_puyos) / 2))
        : cfg.look_ahead;

    for (int depth = 0; depth < effective_look_ahead; ++depth) {
        tl_depth_dedup.advanceDepth();

        int32_t piece_idx = tsumo_base + depth;
        PuyoPiece piece = tsumo.get(piece_idx);
        const bool is_zoro = (piece.axis == piece.sub);
        const bool is_last_depth = (depth == effective_look_ahead - 1);

        tl_candidates.clear();

        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        const int current_size = static_cast<int>(tl_current_beam.size());

        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const auto& parent = tl_current_beam[p_idx];
            const int32_t parent_accum = parent.accum_score();
            const int parent_first_action = parent.first_action();
            const uint64_t parent_hash = parent.hash;
            const uint32_t parent_packed_heights = parent.packed_heights();
            const bool parent_has_fired_main = parent.has_fired_main();

            for (uint8_t a_idx = 0; a_idx < actions.size(); ++a_idx) {
                const auto& entry = actions[a_idx];
                const int h_axis = (parent_packed_heights >> (entry.ax << 2)) & 0xFu;
                const int h_sub  = (parent_packed_heights >> (entry.sx << 2)) & 0xFu;
                const int y_axis = h_axis + entry.axis_dy;
                const int y_sub  = h_sub  + entry.sub_dy;

                PlaceResult pr;
                simulatePlacement(parent.field, piece, entry, parent_packed_heights, pr);

                if (pr.dead)
                    continue;

                uint32_t next_packed_h;
                uint64_t child_hash;
                if (pr.score > 0) {
                    next_packed_h = packHeights(pr.field);
                    child_hash = Zobrist::hashBoard(pr.field);
                } else {
                    const uint32_t add_ax = static_cast<uint32_t>(y_axis < config::Board::kHeight) << (entry.ax << 2);
                    const uint32_t add_sx = static_cast<uint32_t>(y_sub  < config::Board::kHeight) << (entry.sx << 2);
                    next_packed_h = parent_packed_heights + add_ax + add_sx;

                    child_hash = parent_hash ^ Zobrist::xorPuyo(piece.axis, entry.ax, y_axis)
                                             ^ Zobrist::xorPuyo(piece.sub,  entry.sx, y_sub);
                }

                const int32_t step_score = static_cast<int32_t>(pr.score);
                const bool next_has_fired_main = parent_has_fired_main || (cfg.main_chain_threshold > 0 && step_score >= cfg.main_chain_threshold);
                const int32_t next_accum = parent_accum + step_score;

                // 本線大連鎖発火済みの場合はセカンドポテンシャルを加算しない（重計算を完全スキップ）
                int32_t total_score;
                if (next_has_fired_main) {
                    total_score = next_accum * cfg.eval_weights.potential_score_scale;
                } else {
                    int32_t pot_score = 0;
                    if (!tl_tt.get(child_hash, pot_score)) {
                        pot_score = computeMaxPotentialScore(pr.field, next_packed_h);
                        tl_tt.put(child_hash, pot_score);
                    }

                    int32_t eval;
                    if constexpr (std::is_same_v<EvaluatorType, SoloBeamEvaluator>) {
                        eval = EvaluatorType::evaluateWithPotential(pr.field, cfg.eval_weights, next_packed_h, pot_score);
                    } else {
                        eval = EvaluatorType::evaluateWithPotential(pr.field, cfg.eval_weights, next_packed_h, pot_score, &cfg.context);
                    }

                    total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;
                }

                if (is_last_depth) {
                    if (total_score > best_score) {
                        best_score = total_score;
                        best_action = (depth == 0) ? entry.idx : parent_first_action;
                        tl_best_leaf_field = pr.field;
                    }
                } else {
                    tl_candidates.push_back({
                        child_hash,
                        total_score,
                        next_packed_h,
                        static_cast<uint32_t>(p_idx),
                        a_idx,
                        next_has_fired_main,
                        {0, 0}
                    });
                }
            }
        }

        if (is_last_depth) {
            break;
        }

        if (tl_candidates.empty())
            break;

        const int target_beam_width = cfg.target_beam_widths[depth];
        const int keep = std::min(static_cast<int>(tl_candidates.size()), target_beam_width);

        tl_prev_beam.swap(tl_current_beam);
        tl_current_beam.clear();

        auto instantiate_node = [&](const CandidateNode& item) {
            const auto& parent = tl_prev_beam[item.parent_idx];
            const auto& act = actions[item.action_idx];
            
            PlaceResult pr;
            simulatePlacement(parent.field, piece, act, parent.packed_heights(), pr);
            int first = (depth == 0) ? act.idx : parent.first_action();
            int32_t next_accum = parent.accum_score() + static_cast<int32_t>(pr.score);

            tl_current_beam.emplace_back(pr.field, next_accum, first, item.packed_heights, item.hash, item.has_fired_main);
        };

        // --- 候補選択・DBSフィルタリング (動的アダプティブ Top-K 最適化) ---
        const auto candidate_cmp = [](const CandidateNode& a, const CandidateNode& b) noexcept {
            return a.score > b.score;
        };

        if (cfg.dbs_max_similar >= 1) {
            tl_dbs_table.clear();
        }

        const size_t total_cands = tl_candidates.size();
        size_t processed_end     = 0;

        while (static_cast<int>(tl_current_beam.size()) < keep && processed_end < total_cands) {
            // 必要残数 × 2 (下限 2048) で最小限のチャンクサイズを算出
            const size_t needed     = static_cast<size_t>(keep - static_cast<int>(tl_current_beam.size()));
            const size_t chunk_size = std::max<size_t>(needed * 2, 2048);
            const size_t next_end   = std::min(total_cands, processed_end + chunk_size);

            if (next_end < total_cands) {
                std::nth_element(tl_candidates.begin() + processed_end,
                                 tl_candidates.begin() + next_end,
                                 tl_candidates.end(),
                                 candidate_cmp);
            }

            std::sort(tl_candidates.begin() + processed_end,
                      tl_candidates.begin() + next_end,
                      candidate_cmp);

            for (size_t i = processed_end; i < next_end; ++i) {
                const auto& item = tl_candidates[i];
                if (tl_depth_dedup.checkAndInsert(item.hash))
                    continue;

                if (cfg.dbs_max_similar >= 1) {
                    if (tl_dbs_table.get_and_inc(item.packed_heights) >= cfg.dbs_max_similar) {
                        continue;
                    }
                }

                instantiate_node(item);
                if (static_cast<int>(tl_current_beam.size()) == keep) {
                    break;
                }
            }

            processed_end = next_end;
        }
    }

    if (best_action == -1 && !tl_current_beam.empty()) {
        best_action = tl_current_beam[0].first_action();
    }

    if constexpr (HasFireBias) {
        if (fire_best_action >= 0 && best_action >= 0) {
            const int64_t fire_val = (static_cast<int64_t>(fire_best_score) * cfg.eval_weights.fire_bias_permille) / 1000;
            if (fire_val > best_score) {
                return {fire_best_action, fire_best_score};
            }
        }
    }

    if (best_action >= 0)
        return {best_action, best_score};

    return {0, -1000000000};
}

std::pair<int, int32_t> soloBeamSearch(const PuyotanPlayer& player,
                                       const Tsumo& tsumo_const,
                                       const SoloBeamConfig& cfg,
                                       BeamSearchSession* session) noexcept {
    auto res = beamSearchImpl<SoloBeamConfig, SoloBeamEvaluator, false>(player, tsumo_const, cfg);
    if (session) {
        session->update(res.second);
    }
    return res;
}

std::pair<int, int32_t> vsBeamSearch(const PuyotanPlayer& player,
                                     const Tsumo& tsumo_const,
                                     const VsBeamConfig& cfg,
                                     BeamSearchSession* session) noexcept {
    auto res = beamSearchImpl<VsBeamConfig, VsBeamEvaluator, true>(player, tsumo_const, cfg);
    if (session) {
        session->update(res.second);
    }
    return res;
}

Board getBestLeafField() noexcept {
    return tl_best_leaf_field;
}

// =========================================================================
// micro_ply対応 ＋ PVキャッシュ＆計画追従ビームサーチ (PV-Guided micro_ply Beam Search)
// =========================================================================

namespace {
struct NodeTrace {
    uint32_t parent_idx;
    uint8_t  action_idx;
};

// 深さごとの親追跡テーブル（最善葉からのバックトラック用）
thread_local std::vector<std::vector<NodeTrace>> tl_pv_tree_trace;
} // anonymous namespace

std::pair<int, int32_t> soloBeamSearchPV(const PuyotanPlayer& player,
                                        const Tsumo& tsumo_const,
                                        const SoloBeamConfig& cfg,
                                        SoloPvPlan* plan) noexcept {
    const int tsumo_base = player.active_next_pos;
    bool has_valid_plan = false;
    struct ActivePv {
        std::vector<uint8_t> actions;
        int target_parent_idx = 0;
    };
    std::vector<ActivePv> active_pvs;
    int32_t prev_planned_score = 0;
    int prev_planned_first_act = -1;

    if (plan && plan->has_plan()) {
        const int occupied_now = player.field.getOccupied().popcount();
        if (cfg.main_chain_threshold > 0 && occupied_now < 16 && plan->best_score() >= cfg.main_chain_threshold) {
            // 盤面がほぼ更地なのに前回の本線高得点計画が残っている場合はセカンド開始のためリセット
            plan->reset();
        }
    }

    if (cfg.pv_elite_count > 0 && plan && plan->has_plan()) {
        if (plan->planned_tsumo_pos == tsumo_base) {
            has_valid_plan = true;
            const int max_routes = std::min(static_cast<int>(plan->routes.size()), cfg.pv_elite_count);
            for (int r = 0; r < max_routes; ++r) {
                if (!plan->routes[r].actions.empty()) {
                    active_pvs.push_back({ plan->routes[r].actions, 0 });
                }
            }
            if (!active_pvs.empty()) {
                prev_planned_score = plan->routes[0].score;
                prev_planned_first_act = active_pvs[0].actions[0];
            }
        } else {
            plan->reset();
        }
    } else if (plan && cfg.pv_elite_count <= 0) {
        plan->reset();
    }

    tl_tt.advanceGeneration();
    Zobrist::init();

    const Tsumo& tsumo = tsumo_const;
    uint32_t packed_heights_root = packHeights(player.field);
    const uint64_t root_hash = Zobrist::hashBoard(player.field);

    tl_current_beam.clear();
    tl_current_beam.reserve(static_cast<std::size_t>(cfg.beam_width));
    tl_prev_beam.clear();
    tl_prev_beam.reserve(static_cast<std::size_t>(cfg.beam_width));

    tl_candidates.clear();
    tl_candidates.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    if (cfg.dbs_max_similar >= 1) {
        tl_dbs_table.ensure_capacity(static_cast<std::size_t>(cfg.beam_width));
    }

    tl_current_beam.emplace_back(player.field, 0, -1, packed_heights_root, root_hash, false);

    const int occupied_puyos = player.field.getOccupied().popcount();
    const int effective_look_ahead = (cfg.dynamic_lookahead_margin > 0)
        ? std::min(cfg.look_ahead, std::max(2, (cfg.dynamic_lookahead_margin - occupied_puyos) / 2))
        : cfg.look_ahead;

    if (tl_pv_tree_trace.size() < static_cast<size_t>(effective_look_ahead)) {
        tl_pv_tree_trace.resize(effective_look_ahead);
    }
    for (int d = 0; d < effective_look_ahead; ++d) {
        tl_pv_tree_trace[d].clear();
    }

    int best_action = -1;
    int32_t best_score = -1000000000;

    struct BestLeaf {
        uint32_t trace_idx;
        int32_t  score;
        bool     is_pv;
    };
    std::vector<BestLeaf> best_leaves;

    for (int depth = 0; depth < effective_look_ahead; ++depth) {
        tl_depth_dedup.advanceDepth();

        int32_t piece_idx = tsumo_base + depth;
        PuyoPiece piece = tsumo.get(piece_idx);
        const bool is_zoro = (piece.axis == piece.sub);
        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        const bool is_last_depth = (depth == effective_look_ahead - 1);

        tl_candidates.clear();
        const int current_size = static_cast<int>(tl_current_beam.size());

        std::vector<uint8_t> pv_action_for_depth(active_pvs.size(), 255);
        std::vector<int> next_pv_target_parent_idx(active_pvs.size(), -1);
        for (size_t r = 0; r < active_pvs.size(); ++r) {
            if (depth < static_cast<int>(active_pvs[r].actions.size())) {
                pv_action_for_depth[r] = active_pvs[r].actions[depth];
            }
        }

        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const auto& parent = tl_current_beam[p_idx];
            const int32_t parent_accum = parent.accum_score();
            const int parent_first_action = parent.first_action();
            const uint32_t parent_packed_h = parent.packed_heights();
            const bool parent_has_fired_main = parent.has_fired_main();

            int matched_pv_route = -1;
            for (size_t r = 0; r < active_pvs.size(); ++r) {
                if (p_idx == active_pvs[r].target_parent_idx && pv_action_for_depth[r] != 255) {
                    matched_pv_route = static_cast<int>(r);
                    break;
                }
            }

            for (uint8_t a_idx = 0; a_idx < actions.size(); ++a_idx) {
                const auto& entry = actions[a_idx];
                PlaceResult pr;
                simulatePlacement(parent.field, piece, entry, parent_packed_h, pr);
                if (pr.dead) continue;

                uint32_t next_packed_h;
                uint64_t child_hash;
                if (pr.score > 0) {
                    next_packed_h = packHeights(pr.field);
                    child_hash = Zobrist::hashBoard(pr.field);
                } else {
                    const int h_axis = (parent_packed_h >> (entry.ax << 2)) & 0xFu;
                    const int h_sub  = (parent_packed_h >> (entry.sx << 2)) & 0xFu;
                    const int y_axis = h_axis + entry.axis_dy;
                    const int y_sub  = h_sub  + entry.sub_dy;
                    const uint32_t add_ax = static_cast<uint32_t>(y_axis < config::Board::kHeight) << (entry.ax << 2);
                    const uint32_t add_sx = static_cast<uint32_t>(y_sub  < config::Board::kHeight) << (entry.sx << 2);
                    next_packed_h = parent_packed_h + add_ax + add_sx;
                    child_hash = parent.hash ^ Zobrist::xorPuyo(piece.axis, entry.ax, y_axis)
                                             ^ Zobrist::xorPuyo(piece.sub,  entry.sx, y_sub);
                }

                const int32_t step_score = static_cast<int32_t>(pr.score);
                const bool next_has_fired_main = parent_has_fired_main || (cfg.main_chain_threshold > 0 && step_score >= cfg.main_chain_threshold);
                const int32_t next_accum = parent_has_fired_main ? parent_accum : (parent_accum + step_score);

                const int32_t total_score = evaluateMicroPly(
                    pr.field, next_accum, next_packed_h, child_hash,
                    tsumo, tsumo_base + depth + 1,
                    cfg.micro_ply - 1,
                    cfg, next_has_fired_main);

                const int first_act = (depth == 0) ? entry.idx : parent_first_action;
                const bool is_pv_node = (matched_pv_route >= 0) && (entry.idx == pv_action_for_depth[matched_pv_route]);

                if (is_last_depth) {
                    if (total_score > best_score || (is_pv_node && total_score >= best_score)) {
                        best_score = total_score;
                        best_action = first_act;
                        tl_best_leaf_field = pr.field;
                    }
                    const uint32_t leaf_idx = static_cast<uint32_t>(tl_pv_tree_trace[depth].size());
                    tl_pv_tree_trace[depth].push_back({ static_cast<uint32_t>(p_idx), static_cast<uint8_t>(entry.idx) });

                    if (cfg.pv_elite_count > 0) {
                        const int max_leaves = std::clamp(cfg.pv_elite_count, 1, 10);
                        if (static_cast<int>(best_leaves.size()) < max_leaves || total_score > best_leaves.back().score) {
                            best_leaves.push_back({ leaf_idx, total_score, is_pv_node });
                            std::sort(best_leaves.begin(), best_leaves.end(), [](const auto& a, const auto& b) {
                                if (a.score != b.score) return a.score > b.score;
                                return a.is_pv > b.is_pv;
                            });
                            if (static_cast<int>(best_leaves.size()) > max_leaves) {
                                best_leaves.pop_back();
                            }
                        }
                    }
                } else {
                    tl_candidates.push_back({
                        child_hash,
                        total_score,
                        next_packed_h,
                        static_cast<uint32_t>(p_idx),
                        a_idx,
                        next_has_fired_main,
                        { static_cast<uint8_t>(is_pv_node ? (matched_pv_route + 1) : 0), 0 }
                    });
                }
            }
        }

        if (is_last_depth || tl_candidates.empty()) break;

        const int target_beam_width = cfg.target_beam_widths[depth];
        const int keep = std::min(static_cast<int>(tl_candidates.size()), target_beam_width);

        tl_prev_beam.swap(tl_current_beam);
        tl_current_beam.clear();
        tl_pv_tree_trace[depth].clear();

        if (cfg.dbs_max_similar >= 1) {
            tl_dbs_table.clear();
        }

        // スコア降順ソート（同点時は端優先）
        std::sort(tl_candidates.begin(), tl_candidates.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.action_idx < b.action_idx;
        });

        // ★ Phase 1: elite_keep 枠 — DBS を完全スキップして上位N個を無条件保護
        const int elite_n = std::min(cfg.elite_keep, keep);
        int elite_added = 0;
        if (elite_n > 0) {
            for (const auto& item : tl_candidates) {
                if (elite_added >= elite_n) break;
                const bool is_pv = (item._pad[0] != 0);
                if (!is_pv) {
                    if (tl_depth_dedup.checkAndInsert(item.hash)) continue;
                } else {
                    tl_depth_dedup.checkAndInsert(item.hash);
                }

                if (cfg.dbs_max_similar >= 1) {
                    tl_dbs_table.get_and_inc(item.packed_heights);
                }

                const auto& parent = tl_prev_beam[item.parent_idx];
                const auto& act = actions[item.action_idx];
                PlaceResult pr;
                simulatePlacement(parent.field, piece, act, parent.packed_heights(), pr);

                int first = (depth == 0) ? act.idx : parent.first_action();
                int32_t next_accum = parent.accum_score() + (parent.has_fired_main() ? 0 : static_cast<int32_t>(pr.score));

                const uint32_t new_node_idx = static_cast<uint32_t>(tl_current_beam.size());
                tl_current_beam.emplace_back(pr.field, next_accum, first, item.packed_heights, item.hash, item.has_fired_main);
                tl_pv_tree_trace[depth].push_back({ item.parent_idx, static_cast<uint8_t>(act.idx) });

                // PVノードなら追跡インデックスも更新
                if (is_pv) {
                    const int r = static_cast<int>(item._pad[0]) - 1;
                    if (r >= 0 && r < static_cast<int>(next_pv_target_parent_idx.size())) {
                        if (next_pv_target_parent_idx[r] == -1) {
                            next_pv_target_parent_idx[r] = static_cast<int>(new_node_idx);
                        }
                    }
                }

                ++elite_added;
            }
        }

        // ★ Phase 2: 残り枠 — PV elite（DBS スキップ）＋ 通常フィルタ
        for (const auto& item : tl_candidates) {
            if (static_cast<int>(tl_current_beam.size()) >= keep) break;
            const bool is_pv_elite = (item._pad[0] != 0);

            if (!is_pv_elite) {
                if (tl_depth_dedup.checkAndInsert(item.hash)) continue;
                if (cfg.dbs_max_similar >= 1 && tl_dbs_table.get_and_inc(item.packed_heights) >= cfg.dbs_max_similar) {
                    continue;
                }
            } else {
                tl_depth_dedup.checkAndInsert(item.hash);
                if (cfg.dbs_max_similar >= 1) {
                    tl_dbs_table.get_and_inc(item.packed_heights);
                }
            }

            const auto& parent = tl_prev_beam[item.parent_idx];
            const auto& act = actions[item.action_idx];
            PlaceResult pr;
            simulatePlacement(parent.field, piece, act, parent.packed_heights(), pr);

            int first = (depth == 0) ? act.idx : parent.first_action();
            int32_t next_accum = parent.accum_score() + (parent.has_fired_main() ? 0 : static_cast<int32_t>(pr.score));

            const uint32_t new_node_idx = static_cast<uint32_t>(tl_current_beam.size());
            tl_current_beam.emplace_back(pr.field, next_accum, first, item.packed_heights, item.hash, item.has_fired_main);
            tl_pv_tree_trace[depth].push_back({ item.parent_idx, static_cast<uint8_t>(act.idx) });

            if (is_pv_elite) {
                const int r = static_cast<int>(item._pad[0]) - 1;
                if (r >= 0 && r < static_cast<int>(next_pv_target_parent_idx.size())) {
                    if (next_pv_target_parent_idx[r] == -1) {
                        next_pv_target_parent_idx[r] = static_cast<int>(new_node_idx);
                    }
                }
            }
        }

        for (size_t r = 0; r < active_pvs.size(); ++r) {
            active_pvs[r].target_parent_idx = next_pv_target_parent_idx[r];
        }
    }

    // 探索終了：最善葉からのバックトラックで PV（手順列）を復元
    if (cfg.pv_elite_count > 0 && plan && effective_look_ahead > 0 && !tl_pv_tree_trace[effective_look_ahead - 1].empty()) {
        const int max_keep_routes = std::clamp(cfg.pv_elite_count, 1, 10);
        plan->routes.clear();

        for (const auto& leaf : best_leaves) {
            if (static_cast<int>(plan->routes.size()) >= max_keep_routes) break;

            std::vector<uint8_t> new_pv(effective_look_ahead);
            uint32_t cur = leaf.trace_idx;
            if (cur >= tl_pv_tree_trace[effective_look_ahead - 1].size()) {
                cur = static_cast<uint32_t>(tl_pv_tree_trace[effective_look_ahead - 1].size() - 1);
            }

            for (int d = effective_look_ahead - 1; d >= 0; --d) {
                if (cur < tl_pv_tree_trace[d].size()) {
                    new_pv[d] = tl_pv_tree_trace[d][cur].action_idx;
                    cur = tl_pv_tree_trace[d][cur].parent_idx;
                } else {
                    new_pv[d] = 0;
                    cur = 0;
                }
            }

            if (new_pv.size() > 1) {
                std::vector<uint8_t> remaining_actions(new_pv.begin() + 1, new_pv.end());
                bool duplicate = false;
                for (const auto& existing : plan->routes) {
                    if (existing.actions == remaining_actions) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    plan->routes.push_back({ std::move(remaining_actions), leaf.score });
                }
            }
        }
        if (!plan->routes.empty()) {
            plan->planned_tsumo_pos = tsumo_base + 1;
        } else {
            plan->planned_tsumo_pos = -1;
        }
    }

    // 計画コミットメント（スコア後退・浮気の防止）
    int final_action = best_action;
    int32_t final_score = best_score;

    if (cfg.pv_elite_count > 0 && has_valid_plan && prev_planned_first_act >= 0 && best_score < prev_planned_score) {
        final_action = prev_planned_first_act;
        final_score = prev_planned_score;
    }

    if (final_action == -1 && !tl_current_beam.empty()) {
        final_action = tl_current_beam[0].first_action();
    }
    if (final_action < 0) final_action = 0;

    // ★ 本線発火判定: 今回指す手（final_action）で本線が発火するか確認
    bool final_action_fires_main = false;
    if (cfg.main_chain_threshold > 0) {
        int32_t p0_idx = tsumo_base;
        const PuyoPiece p0 = tsumo.get(p0_idx);
        const auto& act_list = (p0.axis == p0.sub) ? getZoroActions() : getPutActions();
        for (const auto& ba : act_list) {
            if (ba.idx == final_action) {
                PlaceResult pr0;
                simulatePlacement(player.field, p0, ba, packed_heights_root, pr0);
                if (!pr0.dead && static_cast<int32_t>(pr0.score) >= cfg.main_chain_threshold) {
                    final_action_fires_main = true;
                }
                break;
            }
        }
    }

    if (final_action_fires_main) {
        // 本線を打った瞬間、それまでの計画は達成完了。
        // 発火後の更地セカンドに古い本線スコアやゴースト手順を持ち越さないようリセット。
        if (plan) {
            plan->reset();
        }
        return { final_action, final_score };
    }

    // 本線未発火時のみ、スコア後退時に前回の計画手順を次回用に引き継ぐ
    if (cfg.pv_elite_count > 0 && has_valid_plan && prev_planned_first_act >= 0 && best_score < prev_planned_score) {
        if (plan && !active_pvs.empty()) {
            plan->routes.clear();
            if (active_pvs[0].actions.size() > 1) {
                std::vector<uint8_t> remaining_actions(active_pvs[0].actions.begin() + 1, active_pvs[0].actions.end());
                plan->routes.push_back({ std::move(remaining_actions), prev_planned_score });
                plan->planned_tsumo_pos = tsumo_base + 1;
            }
        }
        return { final_action, final_score };
    }

    return { final_action, final_score };
}

} // namespace puyotan::search