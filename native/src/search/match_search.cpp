#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/beam_evaluator.hpp>
#include <puyotan/search/match_search.hpp>

namespace puyotan::search {
namespace {

struct MatchNode {
    PuyotanMatch match;
    int32_t score = 0;
    int first_action = -1; // 最初に選んだ自分の手 (0..21)
    int my_depth = 0;      // 自分が手を打った回数
};

// 展開時の候補を記録する軽量構造体 (16 bytes)
struct MatchCandidate {
    int32_t score;
    uint32_t parent_idx;
    uint8_t my_act;       // 0..21 (自分が打たない場合は 255)
    uint8_t enemy_act;    // 0..21 (相手が打たない場合は 255)
    uint8_t is_terminal;  // 1 if terminal or completed
    uint8_t _pad;
};

// 評価関数: 自分目線での優劣スコアを算出
inline int32_t evaluateMatch(const PuyotanMatch& match, int my_id, const MatchBeamConfig& cfg) noexcept {
    const MatchStatus st = match.getStatus();
    const int enemy_id = 1 - my_id;
    const PuyotanPlayer& me = match.getPlayer(my_id);
    const PuyotanPlayer& enemy = match.getPlayer(enemy_id);

    if (st != MatchStatus::Playing) {
        if ((my_id == 0 && st == MatchStatus::WinP1) || (my_id == 1 && st == MatchStatus::WinP2)) {
            // 自分の勝利
            return 10000000 + (me.score - enemy.score);
        } else if ((my_id == 0 && st == MatchStatus::WinP2) || (my_id == 1 && st == MatchStatus::WinP1)) {
            // 自分の敗北
            return -10000000 + (me.score - enemy.score);
        } else {
            // 引き分け
            return -5000000;
        }
    }

    // 盤面形状 & ポテンシャル評価 (VsBeamEvaluator)
    const uint32_t my_h = packHeights(me.field);
    const int32_t my_eval = VsBeamEvaluator::evaluate(me.field, cfg.eval_weights, my_h);

    const uint32_t enemy_h = packHeights(enemy.field);
    const int32_t enemy_eval = VsBeamEvaluator::evaluate(enemy.field, cfg.eval_weights, enemy_h);

    // おじゃまペナルティ & ボーナス (おじゃま1個 = 70点換算)
    const int32_t my_ojama_penalty = static_cast<int32_t>(me.active_ojama) * 140 +
                                     static_cast<int32_t>(me.non_active_ojama) * 70;
    const int32_t enemy_ojama_bonus = static_cast<int32_t>(enemy.active_ojama) * 140 +
                                      static_cast<int32_t>(enemy.non_active_ojama) * 70;

    // スコア差分 + 盤面ポテンシャル差分 + おじゃま差分
    return (me.score - me.used_score) + my_eval - my_ojama_penalty
         - ((enemy.score - enemy.used_score) + enemy_eval - enemy_ojama_bonus);
}

} // namespace

std::pair<int, int32_t> matchBeamSearch(const PuyotanMatch& match,
                                        int my_id,
                                        const MatchBeamConfig& cfg) noexcept {
    const int enemy_id = 1 - my_id;
    const int look_ahead = std::max(1, cfg.look_ahead);

    MatchNode root;
    root.match = match;
    root.first_action = -1;
    root.my_depth = 0;

    // 決定待ち状態でない場合、最初の決定ポイントまで進める
    if (root.match.getStatus() == MatchStatus::Playing && root.match.getDecisionMask() == 0) {
        root.match.stepUntilDecision();
    }

    if (root.match.getStatus() != MatchStatus::Playing) {
        return {-1, 0};
    }

    root.score = evaluateMatch(root.match, my_id, cfg);

    std::vector<MatchNode> current_beam;
    current_beam.push_back(root);

    // 候補ディスクリプタ用バッファ（メモリ再確保を抑制）
    std::vector<MatchCandidate> candidates;
    candidates.reserve(std::min(static_cast<size_t>(cfg.beam_width * 22), size_t{100000}));

    for (int ply = 0; ply < look_ahead; ++ply) {
        candidates.clear();
        const int target_width = (ply < static_cast<int>(cfg.target_beam_widths.size()) && cfg.target_beam_widths[ply] > 0)
                                     ? cfg.target_beam_widths[ply]
                                     : cfg.beam_width;

        bool has_active_nodes = false;

        for (uint32_t p_idx = 0; p_idx < static_cast<uint32_t>(current_beam.size()); ++p_idx) {
            const auto& parent = current_beam[p_idx];

            if (parent.match.getStatus() != MatchStatus::Playing || parent.my_depth >= look_ahead) {
                // 既に決着しているか、指定手数を打ち終えたノードはそのままパススルー
                candidates.push_back(MatchCandidate{
                    .score = parent.score,
                    .parent_idx = p_idx,
                    .my_act = 255,
                    .enemy_act = 255,
                    .is_terminal = 1,
                    ._pad = 0
                });
                continue;
            }

            has_active_nodes = true;
            const int mask = parent.match.getDecisionMask();

            const bool my_turn = (mask & (1 << my_id)) != 0;
            const bool enemy_turn = (mask & (1 << enemy_id)) != 0;

            if (my_turn && enemy_turn) {
                // 両者同時手番: 自分22手 × 相手22手 = 484通り
                for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                    for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(my_id, getRLAction(m_act));
                        sim.setAction(enemy_id, getRLAction(e_act));
                        sim.stepUntilDecision();

                        int32_t sc = evaluateMatch(sim, my_id, cfg);
                        candidates.push_back(MatchCandidate{
                            .score = sc,
                            .parent_idx = p_idx,
                            .my_act = m_act,
                            .enemy_act = e_act,
                            .is_terminal = 0,
                            ._pad = 0
                        });
                    }
                }
            } else if (my_turn) {
                // 自分だけ手番: 22通り
                for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                    PuyotanMatch sim = parent.match;
                    sim.setAction(my_id, getRLAction(m_act));
                    sim.stepUntilDecision();

                    int32_t sc = evaluateMatch(sim, my_id, cfg);
                    candidates.push_back(MatchCandidate{
                        .score = sc,
                        .parent_idx = p_idx,
                        .my_act = m_act,
                        .enemy_act = 255,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                }
            } else if (enemy_turn) {
                // 相手だけ手番: 22通り
                for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                    PuyotanMatch sim = parent.match;
                    sim.setAction(enemy_id, getRLAction(e_act));
                    sim.stepUntilDecision();

                    int32_t sc = evaluateMatch(sim, my_id, cfg);
                    candidates.push_back(MatchCandidate{
                        .score = sc,
                        .parent_idx = p_idx,
                        .my_act = 255,
                        .enemy_act = e_act,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                }
            } else {
                // 決定待ちなし（試合進行中）
                PuyotanMatch sim = parent.match;
                sim.stepUntilDecision();
                int32_t sc = evaluateMatch(sim, my_id, cfg);
                candidates.push_back(MatchCandidate{
                    .score = sc,
                    .parent_idx = p_idx,
                    .my_act = 255,
                    .enemy_act = 255,
                    .is_terminal = 0,
                    ._pad = 0
                });
            }
        }

        if (candidates.empty() || !has_active_nodes) {
            break;
        }

        // Top-K 選定
        const size_t keep_count = std::min(candidates.size(), static_cast<size_t>(target_width));
        if (candidates.size() > keep_count) {
            std::nth_element(candidates.begin(), candidates.begin() + keep_count, candidates.end(),
                             [](const MatchCandidate& a, const MatchCandidate& b) {
                                 return a.score > b.score;
                             });
            candidates.resize(keep_count);
        }

        // 次層のビームノードを再構築
        std::vector<MatchNode> next_beam;
        next_beam.reserve(keep_count);

        for (const auto& cand : candidates) {
            const auto& parent = current_beam[cand.parent_idx];

            if (cand.is_terminal) {
                next_beam.push_back(parent);
                continue;
            }

            MatchNode child;
            child.match = parent.match;
            child.score = cand.score;

            // first_action の決定・引き継ぎ
            if (parent.first_action != -1) {
                child.first_action = parent.first_action;
            } else if (cand.my_act != 255) {
                child.first_action = cand.my_act;
            } else {
                child.first_action = -1;
            }

            child.my_depth = parent.my_depth + (cand.my_act != 255 ? 1 : 0);

            // シミュレーションを実行して子状態を確定
            if (cand.my_act != 255) {
                child.match.setAction(my_id, getRLAction(cand.my_act));
            }
            if (cand.enemy_act != 255) {
                child.match.setAction(enemy_id, getRLAction(cand.enemy_act));
            }
            child.match.stepUntilDecision();

            next_beam.push_back(std::move(child));
        }

        current_beam = std::move(next_beam);
    }

    // 探索完了。最高スコアのノードを探す
    if (current_beam.empty()) {
        return {0, 0};
    }

    auto best_it = std::max_element(current_beam.begin(), current_beam.end(),
                                    [](const MatchNode& a, const MatchNode& b) {
                                        return a.score < b.score;
                                    });

    int best_act = best_it->first_action;
    if (best_act < 0 || best_act >= kNumRLActions) {
        best_act = 0;
    }

    return {best_act, best_it->score};
}

} // namespace puyotan::search
