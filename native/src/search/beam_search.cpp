#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstring>

#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_evaluator.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {
namespace {

struct BeamNode {
    Board field;
    int32_t score;
    int32_t accum_score;
    int first_action;
    uint32_t packed_heights;
};

// 16バイトに収まる超軽量候補記述子
struct alignas(16) CandidateNode {
    int32_t  score;          
    uint32_t packed_heights; 
    uint32_t parent_idx;     
    uint8_t  action_idx;     
    uint8_t  _pad[3];        
};
static_assert(sizeof(CandidateNode) == 16, "CandidateNode must be exactly 16 bytes");

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
// simulatePlacement: 連鎖シミュレーションの極限最適化
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

    if (y_axis >= config::Board::kTotalRows || y_sub >= config::Board::kTotalRows) [[unlikely]] {
        out_res.dead = true;
        return;
    }

    out_res.field = src;
    out_res.chain = 0;
    out_res.score = 0;
    out_res.dead  = false;

    out_res.field.dropPiecePairFast(ax, sx, y_axis, y_sub, piece.axis, piece.sub);

    ErasureData ed;
    Chain::scanGroups(out_res.field, ed, piece.dirty_flag);

    // 連鎖ループ: 誰も落ちていなければ即座に終了する
    while (ed.num_erased > 0) {
        ++out_res.chain;
        out_res.score += Scorer::calculateStepScore(ed, out_res.chain);
        Chain::applyErasure(out_res.field, ed);

        const uint32_t fallen = Gravity::execute(out_res.field);
        // 【最重要最適化】誰も落ちていなければ次の連鎖は絶対に起きないため即脱出
        if (fallen == 0) {
            break;
        }

        Chain::scanGroups(out_res.field, ed, fallen);
    }

    if (out_res.field.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow)) [[unlikely]] {
        out_res.dead = true;
    }
}

} // anonymous namespace

template <typename ConfigType, typename EvaluatorType, bool HasFireBias = false>
std::pair<int, int32_t> beamSearchImpl(const PuyotanPlayer& player,
                                       const Tsumo& tsumo_const,
                                       const ConfigType& cfg) noexcept {
    assert(cfg.dbs_max_similar <= 65535 && "dbs_max_similar must not exceed 65535");

    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    uint32_t packed_heights_root = packHeights(player.field);

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

    tl_current_beam.emplace_back(player.field, 0, 0, -1, packed_heights_root);

    int best_action = -1;
    int32_t best_score = -1000000000;

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        int32_t piece_idx = tsumo_base + depth;
        PuyoPiece piece = tsumo.get(piece_idx);
        const bool is_zoro = (piece.axis == piece.sub);
        const bool is_last_depth = (depth == cfg.look_ahead - 1);

        tl_candidates.clear();

        const auto actions = is_zoro ? getZoroActions() : getPutActions();
        const int current_size = static_cast<int>(tl_current_beam.size());

        PlaceResult pr;

        // Phase 1: 候補の生成と評価
        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const BeamNode& node = tl_current_beam[p_idx];
            const uint32_t cur_heights = node.packed_heights;

            for (uint8_t a_idx = 0; a_idx < static_cast<uint8_t>(actions.size()); ++a_idx) {
                const auto& entry = actions[a_idx];
                
                simulatePlacement(node.field, piece, entry, cur_heights, pr);
                if (pr.dead)
                    continue;

                uint32_t next_packed_h = cur_heights + (1u << (entry.ax << 2)) + (1u << (entry.sx << 2));
                if (__builtin_expect(pr.chain > 0, 0)) {
                    next_packed_h = packHeights(pr.field);
                }

                int32_t eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights, next_packed_h);
                int32_t next_accum = node.accum_score + static_cast<int32_t>(pr.score);
                int32_t total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;

                if (is_last_depth) {
                    if (total_score > best_score) {
                        best_score = total_score;
                        best_action = (depth == 0) ? entry.idx : node.first_action;
                    }
                } else {
                    tl_candidates.push_back({
                        total_score,
                        next_packed_h,
                        static_cast<uint32_t>(p_idx),
                        a_idx,
                        {0, 0, 0}
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

        // Phase 2: 上位候補の選別と実体化
        tl_prev_beam.swap(tl_current_beam);
        tl_current_beam.clear();

        auto instantiate_node = [&](const CandidateNode& item) {
            const auto& parent = tl_prev_beam[item.parent_idx];
            const auto& act = actions[item.action_idx];
            
            simulatePlacement(parent.field, piece, act, parent.packed_heights, pr);
            int first = (depth == 0) ? act.idx : parent.first_action;
            int32_t next_accum = parent.accum_score + static_cast<int32_t>(pr.score);
            
            tl_current_beam.emplace_back(pr.field, item.score, next_accum, first, item.packed_heights);
        };

        if (cfg.dbs_max_similar >= 1) {
            std::sort(tl_candidates.begin(), tl_candidates.end(),
                      [](const CandidateNode& a, const CandidateNode& b) noexcept {
                          return a.score > b.score;
                      });

            tl_dbs_table.clear();
            for (const auto& item : tl_candidates) {
                if (tl_dbs_table.get_and_inc(item.packed_heights) < cfg.dbs_max_similar) {
                    instantiate_node(item);
                    if (static_cast<int>(tl_current_beam.size()) == keep) {
                        break;
                    }
                }
            }
        } else {
            if (keep < static_cast<int>(tl_candidates.size())) {
                std::nth_element(tl_candidates.begin(), tl_candidates.begin() + keep,
                                 tl_candidates.end(),
                                 [](const CandidateNode& a, const CandidateNode& b) noexcept {
                                     return a.score > b.score;
                                 });
            }

            std::sort(tl_candidates.begin(), tl_candidates.begin() + keep,
                      [](const CandidateNode& a, const CandidateNode& b) noexcept {
                          return a.score > b.score;
                      });

            for (int i = 0; i < keep; ++i) {
                instantiate_node(tl_candidates[i]);
            }
        }
    }

    if (best_action == -1 && !tl_current_beam.empty()) {
        best_action = tl_current_beam[0].first_action;
        best_score = tl_current_beam[0].score;
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

} // namespace puyotan::search