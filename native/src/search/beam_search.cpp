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
// 2手1組マクロステップ探索 (2-Ply Macro Beam Search)
// =========================================================================

// 2手分の候補記述子
struct CandidateNode2Ply {
    uint64_t hash;
    int32_t  score;
    uint32_t packed_heights;
    uint32_t parent_idx;
    uint8_t  act1_idx;
    uint8_t  act2_idx;
    bool     has_fired_main;
};

thread_local std::vector<CandidateNode2Ply> tl_candidates_2ply;

// =========================================================================
// 毎深さ・スライディング2手先読みビームサーチ (Sliding 2-Ply Lookahead)
// =========================================================================

std::pair<int, int32_t> soloBeamSearchSliding2Ply(const PuyotanPlayer& player,
                                                 const Tsumo& tsumo_const,
                                                 const SoloBeamConfig& cfg_orig) noexcept {
    // 毎深さ 484 展開するため、ビーム幅は 2,000 前後が最適 (1深さあたり約 96万候補)
    SoloBeamConfig cfg = cfg_orig;

    tl_tt.advanceGeneration();
    Zobrist::init();

    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

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

    int best_action = -1;
    int32_t best_score = -1000000000;

    const int occupied_puyos = player.field.getOccupied().popcount();
    const int effective_look_ahead = (cfg.dynamic_lookahead_margin > 0)
        ? std::min(cfg.look_ahead, std::max(2, (cfg.dynamic_lookahead_margin - occupied_puyos) / 2))
        : cfg.look_ahead;

    // ★ 毎深さ（depth++）確実に進める！
    for (int depth = 0; depth < effective_look_ahead; ++depth) {
        tl_depth_dedup.advanceDepth();

        int32_t idx1 = tsumo_base + depth;
        PuyoPiece piece1 = tsumo.get(idx1);
        const bool has_next_piece = (depth + 1 < effective_look_ahead);
        int32_t idx2 = tsumo_base + depth + 1;
        PuyoPiece piece2 = has_next_piece ? tsumo.get(idx2) : piece1;

        const auto& actions1 = (piece1.axis == piece1.sub) ? getZoroActions() : getPutActions();
        const auto& actions2 = (piece2.axis == piece2.sub) ? getZoroActions() : getPutActions();

        const bool is_last_depth = (depth == effective_look_ahead - 1);
        tl_candidates.clear();

        const int current_size = static_cast<int>(tl_current_beam.size());

        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const auto& parent = tl_current_beam[p_idx];
            const int32_t parent_accum = parent.accum_score();
            const int parent_first_action = parent.first_action();
            const uint32_t parent_packed_h = parent.packed_heights();
            const bool parent_has_fired_main = parent.has_fired_main();

            // --- 1手目の展開 (最大22手) ---
            for (uint8_t a1 = 0; a1 < actions1.size(); ++a1) {
                const auto& act1 = actions1[a1];
                PlaceResult pr1;
                simulatePlacement(parent.field, piece1, act1, parent_packed_h, pr1);
                if (pr1.dead) continue;

                const uint32_t h1_packed = packHeights(pr1.field);
                const uint64_t h1_hash   = Zobrist::hashBoard(pr1.field);
                const int32_t accum1     = parent_accum + static_cast<int32_t>(pr1.score);
                const bool fired1        = parent_has_fired_main || (cfg.main_chain_threshold > 0 && pr1.score >= cfg.main_chain_threshold);

                int first_act = (depth == 0) ? act1.idx : parent_first_action;

                int32_t best_eval_from_here = -1000000000;
                uint32_t best_h2_packed = h1_packed;

                if (!has_next_piece || fired1) {
                    // 最終手または本線発火済みの場合は1手評価
                    int32_t pot = fired1 ? 0 : computeMaxPotentialScore(pr1.field, h1_packed);
                    best_eval_from_here = accum1 * cfg.eval_weights.potential_score_scale + pot;
                    best_h2_packed = h1_packed;
                } else {
                    // ★ 2手目を全展開（最大22手）して、最も良い未来（2手先）のスコアを探索！
                    for (uint8_t a2 = 0; a2 < actions2.size(); ++a2) {
                        const auto& act2 = actions2[a2];
                        PlaceResult pr2;
                        simulatePlacement(pr1.field, piece2, act2, h1_packed, pr2);
                        if (pr2.dead) continue;

                        const uint32_t h2_packed = packHeights(pr2.field);
                        const uint64_t h2_hash   = Zobrist::hashBoard(pr2.field);
                        const int32_t accum2     = accum1 + static_cast<int32_t>(pr2.score);

                        int32_t pot2 = 0;
                        if (!tl_tt.get(h2_hash, pot2)) {
                            pot2 = computeMaxPotentialScore(pr2.field, h2_packed);
                            tl_tt.put(h2_hash, pot2);
                        }

                        int32_t total2 = accum2 * cfg.eval_weights.potential_score_scale + pot2;
                        if (total2 > best_eval_from_here) {
                            best_eval_from_here = total2;
                            best_h2_packed = h2_packed; // ★ 2手後の到達高さを記録
                        }
                    }
                }

                if (best_eval_from_here == -1000000000) continue; // 2手目が全滅した手は除外

                if (is_last_depth) {
                    if (best_eval_from_here > best_score) {
                        best_score = best_eval_from_here;
                        best_action = first_act;
                        tl_best_leaf_field = pr1.field;
                    }
                } else {
                    // ★ 深さ d+1 の候補として「1手目の盤面（act1）」を登録！
                    // DBS判定には「2手後の最善局面の高さプロファイル（best_h2_packed）」を適用！
                    tl_candidates.push_back({
                        h1_hash,
                        best_eval_from_here,
                        best_h2_packed,
                        static_cast<uint32_t>(p_idx),
                        a1,
                        fired1,
                        {0, 0}
                    });
                }
            }
        }

        if (is_last_depth || tl_candidates.empty()) break;

        // --- 次世代ビーム（深さ d+1）の選定 ---
        const int keep = std::min(static_cast<int>(tl_candidates.size()), cfg.beam_width);

        tl_prev_beam.swap(tl_current_beam);
        tl_current_beam.clear();

        if (cfg.dbs_max_similar >= 1) {
            tl_dbs_table.clear();
        }

        std::sort(tl_candidates.begin(), tl_candidates.end(), [](const auto& a, const auto& b) {
            return a.score > b.score;
        });

        // ★ Phase 1: エリート枠 — DBS/dedup を完全スキップして上位N個を無条件保護
        const int elite_n = std::min(cfg.elite_keep, keep);
        int elite_added = 0;
        if (elite_n > 0) {
            for (const auto& item : tl_candidates) {
                if (elite_added >= elite_n) break;

                // エリート枠は重複（同一hash）のみ弾く（盤面が全く同じなら追加する意味がないため）
                // DBS はスキップ → 形が似ていても最高スコアを保護
                if (tl_depth_dedup.checkAndInsert(item.hash)) continue;

                const auto& parent = tl_prev_beam[item.parent_idx];
                PlaceResult pr1;
                simulatePlacement(parent.field, piece1, actions1[item.action_idx], parent.packed_heights(), pr1);

                int first = (depth == 0) ? actions1[item.action_idx].idx : parent.first_action();
                int32_t next_accum = parent.accum_score() + pr1.score;
                const uint32_t real_h1_packed = packHeights(pr1.field);

                // DBSテーブルに登録（後続の通常枠と整合させるため）
                if (cfg.dbs_max_similar >= 1) {
                    tl_dbs_table.get_and_inc(item.packed_heights);
                }

                tl_current_beam.emplace_back(pr1.field, next_accum, first, real_h1_packed, item.hash, item.has_fired_main);
                ++elite_added;
            }
        }

        // ★ Phase 2: 通常枠 — DBS/dedup フィルタを通して残りを埋める
        for (const auto& item : tl_candidates) {
            if (static_cast<int>(tl_current_beam.size()) >= keep) break;
            if (tl_depth_dedup.checkAndInsert(item.hash)) continue;
            if (cfg.dbs_max_similar >= 1 && tl_dbs_table.get_and_inc(item.packed_heights) >= cfg.dbs_max_similar) continue;

            // 1手進めた盤面を深さ d+1 のビームノードとして生成
            const auto& parent = tl_prev_beam[item.parent_idx];
            PlaceResult pr1;
            simulatePlacement(parent.field, piece1, actions1[item.action_idx], parent.packed_heights(), pr1);

            int first = (depth == 0) ? actions1[item.action_idx].idx : parent.first_action();
            int32_t next_accum = parent.accum_score() + pr1.score;

            // ★ ビームノードに持たせる高さは、生成された pr1.field の真の高さ
            const uint32_t real_h1_packed = packHeights(pr1.field);
            tl_current_beam.emplace_back(pr1.field, next_accum, first, real_h1_packed, item.hash, item.has_fired_main);
        }
    }

    if (best_action == -1 && !tl_current_beam.empty()) {
        best_action = tl_current_beam[0].first_action();
    }

    return {best_action >= 0 ? best_action : 0, best_score};
}

// =========================================================================
// 1手完全先読み ＋ PVキャッシュ＆計画追従ビームサーチ (PV-Guided 1-Ply Beam Search)
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
    std::vector<uint8_t> active_pv_actions;
    int32_t prev_planned_score = 0;
    int prev_planned_first_act = -1;

    if (plan && plan->has_plan()) {
        if (plan->planned_tsumo_pos == tsumo_base) {
            has_valid_plan = true;
            if (!plan->routes.empty() && !plan->routes[0].actions.empty()) {
                active_pv_actions = plan->routes[0].actions;
                prev_planned_score = plan->routes[0].score;
                prev_planned_first_act = active_pv_actions[0];
            }
        } else {
            plan->reset();
        }
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
    uint32_t best_leaf_idx = 0;

    int pv_target_parent_idx = 0;

    for (int depth = 0; depth < effective_look_ahead; ++depth) {
        tl_depth_dedup.advanceDepth();

        int32_t piece_idx = tsumo_base + depth;
        PuyoPiece piece = tsumo.get(piece_idx);
        const bool is_zoro = (piece.axis == piece.sub);
        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        const bool is_last_depth = (depth == effective_look_ahead - 1);

        tl_candidates.clear();
        const int current_size = static_cast<int>(tl_current_beam.size());

        const bool has_pv_for_depth = has_valid_plan && (depth < static_cast<int>(active_pv_actions.size()));
        const uint8_t pv_action_for_depth = has_pv_for_depth ? active_pv_actions[depth] : 255;
        int next_pv_target_parent_idx = -1;

        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const auto& parent = tl_current_beam[p_idx];
            const int32_t parent_accum = parent.accum_score();
            const int parent_first_action = parent.first_action();
            const uint32_t parent_packed_h = parent.packed_heights();
            const bool parent_has_fired_main = parent.has_fired_main();

            const bool is_pv_parent = has_pv_for_depth && (p_idx == pv_target_parent_idx);

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
                const int32_t next_accum = parent_accum + step_score;

                int32_t total_score;
                if (next_has_fired_main) {
                    total_score = next_accum * cfg.eval_weights.potential_score_scale;
                } else {
                    int32_t pot_score = 0;
                    if (!tl_tt.get(child_hash, pot_score)) {
                        pot_score = computeMaxPotentialScore(pr.field, next_packed_h);
                        tl_tt.put(child_hash, pot_score);
                    }
                    total_score = next_accum * cfg.eval_weights.potential_score_scale + pot_score;
                }

                const int first_act = (depth == 0) ? entry.idx : parent_first_action;
                const bool is_pv_node = is_pv_parent && (entry.idx == pv_action_for_depth);

                if (is_last_depth) {
                    if (total_score > best_score || (is_pv_node && total_score >= best_score)) {
                        best_score = total_score;
                        best_action = first_act;
                        tl_best_leaf_field = pr.field;
                        best_leaf_idx = static_cast<uint32_t>(tl_pv_tree_trace[depth].size());
                    }
                    tl_pv_tree_trace[depth].push_back({ static_cast<uint32_t>(p_idx), static_cast<uint8_t>(entry.idx) });
                } else {
                    tl_candidates.push_back({
                        child_hash,
                        total_score,
                        next_packed_h,
                        static_cast<uint32_t>(p_idx),
                        a_idx,
                        next_has_fired_main,
                        { static_cast<uint8_t>(is_pv_node ? 1 : 0), 0 }
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
                int32_t next_accum = parent.accum_score() + static_cast<int32_t>(pr.score);

                const uint32_t new_node_idx = static_cast<uint32_t>(tl_current_beam.size());
                tl_current_beam.emplace_back(pr.field, next_accum, first, item.packed_heights, item.hash, item.has_fired_main);
                tl_pv_tree_trace[depth].push_back({ item.parent_idx, static_cast<uint8_t>(act.idx) });

                // PVノードなら追跡インデックスも更新
                if (is_pv) {
                    next_pv_target_parent_idx = static_cast<int>(new_node_idx);
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
            int32_t next_accum = parent.accum_score() + static_cast<int32_t>(pr.score);

            const uint32_t new_node_idx = static_cast<uint32_t>(tl_current_beam.size());
            tl_current_beam.emplace_back(pr.field, next_accum, first, item.packed_heights, item.hash, item.has_fired_main);
            tl_pv_tree_trace[depth].push_back({ item.parent_idx, static_cast<uint8_t>(act.idx) });

            if (is_pv_elite) {
                next_pv_target_parent_idx = static_cast<int>(new_node_idx);
            }
        }

        pv_target_parent_idx = next_pv_target_parent_idx;
    }

    // 探索終了：最善葉からのバックトラックで PV（手順列）を復元
    if (plan && effective_look_ahead > 0 && !tl_pv_tree_trace[effective_look_ahead - 1].empty()) {
        std::vector<uint8_t> new_pv(effective_look_ahead);
        uint32_t cur = best_leaf_idx;
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

        // 次回用の計画として保存（1手先 = tsumo_base + 1 からの手順）
        plan->routes.clear();
        if (new_pv.size() > 1) {
            std::vector<uint8_t> remaining_actions(new_pv.begin() + 1, new_pv.end());
            plan->routes.push_back({ std::move(remaining_actions), best_score });
            plan->planned_tsumo_pos = tsumo_base + 1;
        }
    }

    // 計画コミットメント（スコア後退・浮気の防止）
    if (has_valid_plan && prev_planned_first_act >= 0 && best_score < prev_planned_score) {
        // ★重要: 前回の計画を採用するなら、次回用キャッシュも前回計画の続きを維持する
        if (plan) {
            plan->routes.clear();
            if (active_pv_actions.size() > 1) {
                std::vector<uint8_t> remaining_actions(active_pv_actions.begin() + 1, active_pv_actions.end());
                plan->routes.push_back({ std::move(remaining_actions), prev_planned_score });
                plan->planned_tsumo_pos = tsumo_base + 1;
            }
        }
        return { prev_planned_first_act, prev_planned_score };
    }

    if (best_action == -1 && !tl_current_beam.empty()) {
        best_action = tl_current_beam[0].first_action();
    }

    return { best_action >= 0 ? best_action : 0, best_score };
}

} // namespace puyotan::search