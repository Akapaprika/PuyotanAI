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

struct ScoreIdx {
    int32_t score;
    uint32_t packed_heights;
    int idx;
};

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

thread_local std::vector<ScoreIdx>    tl_sort_buf;
thread_local std::vector<BeamNode>    tl_current_beam;
thread_local std::vector<BeamNode>    tl_next_beam;
thread_local DynamicFlatCountTable    tl_dbs_table;

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

    // 【修正】正しい境界チェック（元の判定に完全復元）
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

    tl_next_beam.clear();
    tl_next_beam.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    tl_sort_buf.clear();
    tl_sort_buf.reserve(static_cast<std::size_t>(cfg.beam_width) * kNumRLActions);

    if (cfg.dbs_max_similar >= 1) {
        tl_dbs_table.ensure_capacity(static_cast<std::size_t>(cfg.beam_width));
    }

    tl_current_beam.emplace_back(player.field, 0, 0, -1, packed_heights_root);

    for (int depth = 0; depth < cfg.look_ahead; ++depth) {
        PuyoPiece piece = tsumo.get(tsumo_base + depth);
        const bool is_zoro = (piece.axis == piece.sub);
        
        tl_next_beam.clear();
        tl_sort_buf.clear();

        const auto& actions = is_zoro ? getZoroActions() : getPutActions();
        for (const BeamNode& node : tl_current_beam) {
            const uint32_t cur_heights = node.packed_heights;

            for (const auto& entry : actions) {
                PlaceResult pr = simulatePlacement(node.field, piece, entry, cur_heights);
                if (pr.dead)
                    continue;

                int32_t eval = EvaluatorType::evaluate(pr.field, cfg.eval_weights);
                int32_t next_accum = node.accum_score + static_cast<int32_t>(pr.score);
                int32_t total_score = next_accum * cfg.eval_weights.potential_score_scale + eval;
                int first = (depth == 0) ? entry.idx : node.first_action;

                uint32_t next_packed_h = packHeights(pr.field);
                int next_idx = static_cast<int>(tl_next_beam.size());

                tl_next_beam.emplace_back(pr.field, total_score, next_accum, first, next_packed_h);
                tl_sort_buf.push_back({total_score, next_packed_h, next_idx});
            }
        }

        if (tl_next_beam.empty())
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

        const int keep = std::min(static_cast<int>(tl_next_beam.size()), target_beam_width);

        if (cfg.dbs_max_similar >= 1) {
            std::sort(tl_sort_buf.begin(), tl_sort_buf.end(),
                      [](const ScoreIdx& a, const ScoreIdx& b) noexcept {
                          return a.score > b.score;
                      });

            tl_dbs_table.clear();

            tl_current_beam.clear();
            for (const auto& item : tl_sort_buf) {
                if (tl_dbs_table.get_and_inc(item.packed_heights) < cfg.dbs_max_similar) {
                    tl_current_beam.push_back(std::move(tl_next_beam[item.idx]));
                    if (static_cast<int>(tl_current_beam.size()) == keep) {
                        break;
                    }
                }
            }
        } else {
            if (keep < static_cast<int>(tl_sort_buf.size())) {
                std::nth_element(tl_sort_buf.begin(), tl_sort_buf.begin() + keep,
                                 tl_sort_buf.end(),
                                 [](const ScoreIdx& a, const ScoreIdx& b) noexcept {
                                     return a.score > b.score;
                                 });
            }

            std::sort(tl_sort_buf.begin(), tl_sort_buf.begin() + keep,
                      [](const ScoreIdx& a, const ScoreIdx& b) noexcept {
                          return a.score > b.score;
                      });

            tl_current_beam.clear();
            for (int i = 0; i < keep; ++i) {
                tl_current_beam.push_back(std::move(tl_next_beam[tl_sort_buf[i].idx]));
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