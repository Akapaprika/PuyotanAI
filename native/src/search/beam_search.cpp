#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <puyotan/common/types.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/action_table.hpp>
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

// 16バイトに収まる軽量候補記述子 (AMD 3020e の L2 キャッシュ 512KB に余裕で収まる)
struct alignas(16) CandidateNode {
    int32_t score;
    int32_t accum_score;
    uint32_t packed_heights;
    
    // ★ 4バイト(32bit)の空間をビット単位で分割し、16バイト境界を死守する
    uint32_t parent_idx : 24; // 最大 16,777,215 まで許容 (40000でも全く問題なし)
    uint32_t action_idx : 7;  // 0〜127 (必要なのは0〜21)
    uint32_t is_zoro    : 1;  // 0〜1 (フラグ)
};
static_assert(sizeof(CandidateNode) == 16, "CandidateNode must be exactly 16 bytes");

// ---------------------------------------------------------------------------
// DynamicFlatCountTable: beam_width に応じて安全に拡張するフラットハッシュ
// ---------------------------------------------------------------------------
struct DynamicFlatCountTable {
    struct Entry {
        uint32_t key;
        int count;
        uint32_t gen;
    };
    std::vector<Entry> table;
    uint32_t mask = 0;
    uint32_t current_gen = 1;

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
        // フィボナッチハッシュで全スロットに均等分散
        std::size_t idx = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) >> 32 & mask;
        while (table[idx].gen == current_gen) {
            if (table[idx].key == key) {
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

PlaceResult simulatePlacement(const Board& src, PuyoPiece piece,
                              const BeamAction& action,
                              uint32_t packed_heights) noexcept {
    const int ax = action.ax;
    const int sx = action.sx;

    const int h_axis = (packed_heights >> (ax << 2)) & 0xFu;
    const int h_sub  = (packed_heights >> (sx << 2)) & 0xFu;

    const int y_axis = h_axis + action.axis_dy;
    const int y_sub  = h_sub + action.sub_dy;

    if (y_axis >= config::Board::kTotalRows || y_sub >= config::Board::kTotalRows) [[unlikely]] {
        return {Board{}, 0, 0, true};
    }

    PlaceResult res{src, 0, 0, false};
    res.field.dropPiecePairFast(ax, sx, y_axis, y_sub, piece.axis, piece.sub);

    // 連鎖解決
    ErasureData ed;
    Chain::scanGroups(res.field, ed, piece.dirty_flag);
    while (ed.num_erased > 0) {
        ++res.chain;
        res.score += Scorer::calculateStepScore(ed, res.chain);
        Chain::applyErasure(res.field, ed);

        uint32_t fallen = Gravity::execute(res.field);
        Chain::scanGroups(res.field, ed, fallen);
    }

    if (res.field.isOccupied(config::Rule::kDeathCol, config::Rule::kDeathRow)) [[unlikely]] {
        res.dead = true;
        return res;
    }

    return res;
}

} // anonymous namespace

template <typename ConfigType, typename EvaluatorType, bool HasFireBias = false>
std::pair<int, int32_t> beamSearchImpl(const PuyotanPlayer& player,
                                       const Tsumo& tsumo_const,
                                       const ConfigType& cfg) noexcept {
    const Tsumo& tsumo = tsumo_const;
    const int tsumo_base = player.active_next_pos;

    uint32_t packed_heights_root = packHeights(player.field);

    int fire_best_action = -1;
    int32_t fire_best_score = 0;
    if constexpr (HasFireBias) {
        PuyoPiece piece0 = tsumo.get(tsumo_base + 0);
        const bool is_zoro0 = (piece0.axis == piece0.sub);
        const auto& actions0 = is_zoro0 ? getZoroActions() : getPutActions();
        for (const auto& entry : actions0) {
            PlaceResult pr = simulatePlacement(player.field, piece0, entry, packed_heights_root);
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

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        PuyoPiece piece = tsumo.get(tsumo_base + depth);
        const bool is_zoro = (piece.axis == piece.sub);
        
        tl_candidates.clear();

        const auto actions = is_zoro ? getZoroActions() : getPutActions();
        const int current_size = static_cast<int>(tl_current_beam.size());

        // Phase 1: 候補の生成と評価（盤面は保存せず 16B の候補記述子のみ収集）
        for (int p_idx = 0; p_idx < current_size; ++p_idx) {
            const BeamNode& node = tl_current_beam[p_idx];
            const uint32_t cur_heights = node.packed_heights;

            for (uint8_t a_idx = 0; a_idx < static_cast<uint8_t>(actions.size()); ++a_idx) {
                const auto& entry = actions[a_idx];
                PlaceResult pr = simulatePlacement(node.field, piece, entry, cur_heights);
                if (pr.dead)
                    continue;

                int32_t eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights);
                int32_t next_accum = node.accum_score + static_cast<int32_t>(pr.score);
                int32_t total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;
                uint32_t next_packed_h = packHeights(pr.field);

                tl_candidates.push_back(CandidateNode{
                    total_score,
                    next_accum,
                    next_packed_h,
                    static_cast<uint32_t>(p_idx),
                    a_idx,
                    static_cast<uint32_t>(is_zoro)
                });
            }
        }

        if (tl_candidates.empty())
            break;

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

        const int keep = std::min(static_cast<int>(tl_candidates.size()), target_beam_width);

        // Phase 2: 上位候補の選別と、勝ち残った盤面のみの実体化
        if (cfg.dbs_max_similar >= 1) {
            std::sort(tl_candidates.begin(), tl_candidates.end(),
                      [](const CandidateNode& a, const CandidateNode& b) noexcept {
                          return a.score > b.score;
                      });

            tl_dbs_table.clear();

            tl_prev_beam.swap(tl_current_beam);
            tl_current_beam.clear();

            for (const auto& item : tl_candidates) {
                if (tl_dbs_table.get_and_inc(item.packed_heights) < cfg.dbs_max_similar) {
                    const auto& parent = tl_prev_beam[item.parent_idx];
                    const auto act_list = item.is_zoro ? getZoroActions() : getPutActions();
                    const auto& act = act_list[item.action_idx];
                    
                    PlaceResult pr = simulatePlacement(parent.field, piece, act, parent.packed_heights);
                    int first = (depth == 0) ? act.idx : parent.first_action;
                    
                    tl_current_beam.emplace_back(pr.field, item.score, item.accum_score, first, item.packed_heights);
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

            tl_prev_beam.swap(tl_current_beam);
            tl_current_beam.clear();

            for (int i = 0; i < keep; ++i) {
                const auto& item = tl_candidates[i];
                const auto& parent = tl_prev_beam[item.parent_idx];
                const auto act_list = item.is_zoro ? getZoroActions() : getPutActions();
                const auto& act = act_list[item.action_idx];
                
                PlaceResult pr = simulatePlacement(parent.field, piece, act, parent.packed_heights);
                int first = (depth == 0) ? act.idx : parent.first_action;
                
                tl_current_beam.emplace_back(pr.field, item.score, item.accum_score, first, item.packed_heights);
            }
        }
    }

    if constexpr (HasFireBias) {
        if (fire_best_action >= 0 && !tl_current_beam.empty()) {
            const int32_t beam_score = tl_current_beam[0].score;
            const int64_t fire_val = (static_cast<int64_t>(fire_best_score) * cfg.eval_weights.fire_bias_permille) / 1000;
            if (fire_val > beam_score) {
                return {fire_best_action, fire_best_score};
            }
        }
    }

    if (!tl_current_beam.empty() && tl_current_beam[0].first_action >= 0)
        return {tl_current_beam[0].first_action, tl_current_beam[0].score};

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