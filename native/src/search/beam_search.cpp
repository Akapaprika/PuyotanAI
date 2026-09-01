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

                const uint32_t next_packed_h = packHeights(pr.field);

                uint64_t child_hash;
                if (pr.score > 0 || y_axis >= config::Board::kSpawnRow || y_sub >= config::Board::kSpawnRow) {
                    child_hash = Zobrist::hashBoard(pr.field);
                } else {
                    child_hash = parent_hash;
                    if (y_axis < config::Board::kHeight) {
                        child_hash ^= Zobrist::xorPuyo(piece.axis, entry.ax, y_axis);
                    }
                    if (y_sub < config::Board::kHeight) {
                        child_hash ^= Zobrist::xorPuyo(piece.sub, entry.sx, y_sub);
                    }
                }

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

                const int32_t step_score = static_cast<int32_t>(pr.score);
                const bool next_has_fired_main = parent_has_fired_main || (cfg.main_chain_threshold > 0 && step_score >= cfg.main_chain_threshold);
                const int32_t next_accum = parent_accum + step_score;

                // 本線大連鎖発火済みの場合はセカンドポテンシャルを加算しない（水増し防止）
                int32_t total_score;
                if (next_has_fired_main) {
                    total_score = next_accum * cfg.eval_weights.potential_score_scale;
                } else {
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

} // namespace puyotan::search