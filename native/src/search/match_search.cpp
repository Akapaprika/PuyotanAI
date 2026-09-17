#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <immintrin.h>
#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/match_evaluator.hpp>
#include <puyotan/search/match_search.hpp>
#include <puyotan/search/opponent_model.hpp>
#include <puyotan/search/potential_score.hpp>

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

    root.score = MatchBeamEvaluator::evaluate(root.match, my_id, cfg);

    std::vector<MatchNode> current_beam;
    current_beam.push_back(root);

    // 候補ディスクリプタ用バッファ（メモリ再確保を抑制）
    std::vector<MatchCandidate> candidates;
    candidates.reserve(std::min(static_cast<size_t>(cfg.beam_width * 22), size_t{100000}));

    const int min_turns = look_ahead;
    const int max_turns = std::max(min_turns + 25, 35);

    for (int ply = 0; ply < max_turns; ++ply) {
        candidates.clear();
        const int target_width = (ply < static_cast<int>(cfg.target_beam_widths.size()) && cfg.target_beam_widths[ply] > 0)
                                     ? cfg.target_beam_widths[ply]
                                     : cfg.beam_width;

        bool has_active_nodes = false;

        for (uint32_t p_idx = 0; p_idx < static_cast<uint32_t>(current_beam.size()); ++p_idx) {
            const auto& parent = current_beam[p_idx];

            // 終了判定:
            // 1. 勝敗が決着している (WinP1, WinP2, Draw)
            // 2. 最低ターン数 (min_turns) 以上経過しており、かつ盤面が完全に静穏化（連鎖・おじゃま完了）している
            if (parent.match.getStatus() != MatchStatus::Playing ||
                (ply >= min_turns && MatchBeamEvaluator::isMatchQuiescent(parent.match))) {
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
            const bool is_extension = (ply >= min_turns);

            if (is_extension) {
                // ────────────────────────────────────────────────────────────
                // ターン延長フェーズ:
                // 連鎖やおじゃまの決着を見届けるため、新規着手分岐は行わずシミュレーションを進める
                // ────────────────────────────────────────────────────────────
                if (my_turn && enemy_turn) {
                    // 自分は待機 (Pass)，相手は Min 手 (評価を最も下げる手)
                    uint8_t worst_e_act = 0;
                    int32_t worst_sc = 2000000000;
                    for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(my_id, Action{ActionType::Pass});
                        sim.setAction(enemy_id, getRLAction(e_act));
                        sim.stepUntilDecision();

                        int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                        if (sc < worst_sc) {
                            worst_sc = sc;
                            worst_e_act = e_act;
                        }
                    }
                    candidates.push_back(MatchCandidate{
                        .score = worst_sc,
                        .parent_idx = p_idx,
                        .my_act = 254, // 254 = Pass
                        .enemy_act = worst_e_act,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                } else if (my_turn) {
                    // 自分だけ手番: Pass して相手の連鎖・おじゃま落下を待つ
                    PuyotanMatch sim = parent.match;
                    sim.setAction(my_id, Action{ActionType::Pass});
                    sim.stepUntilDecision();
                    int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                    candidates.push_back(MatchCandidate{
                        .score = sc,
                        .parent_idx = p_idx,
                        .my_act = 254, // 254 = Pass
                        .enemy_act = 255,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                } else if (enemy_turn) {
                    // 相手だけ手番: 相手は Min 手
                    uint8_t worst_e_act = 0;
                    int32_t worst_sc = 2000000000;
                    for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(enemy_id, getRLAction(e_act));
                        sim.stepUntilDecision();

                        int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                        if (sc < worst_sc) {
                            worst_sc = sc;
                            worst_e_act = e_act;
                        }
                    }
                    candidates.push_back(MatchCandidate{
                        .score = worst_sc,
                        .parent_idx = p_idx,
                        .my_act = 255,
                        .enemy_act = worst_e_act,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                } else {
                    // 決定待ちなし（試合進行中）
                    PuyotanMatch sim = parent.match;
                    sim.stepUntilDecision();
                    int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                    candidates.push_back(MatchCandidate{
                        .score = sc,
                        .parent_idx = p_idx,
                        .my_act = 255,
                        .enemy_act = 255,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                }
            } else {
                // ────────────────────────────────────────────────────────────
                // 通常ターンフェーズ (ply < min_turns)
                // ────────────────────────────────────────────────────────────
                if (my_turn && enemy_turn) {
                    // 相手が脅威（被攻撃・保留おじゃま・自連鎖）に晒されているか判定
                    const bool enemy_threatened = OpponentModel::isThreatened(parent.match, enemy_id);

                    if (!enemy_threatened) {
                        // 平時ビルドアップモード:
                        // 相手も連鎖を伸ばすプレイヤーとして最善の積み手 (enemy_build_act) を打つ
                        const uint8_t enemy_build_act = OpponentModel::selectBuildMove(parent.match, enemy_id, cfg);

                        for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                            PuyotanMatch sim = parent.match;
                            sim.setAction(my_id, getRLAction(m_act));
                            sim.setAction(enemy_id, getRLAction(enemy_build_act));
                            sim.stepUntilDecision();

                            int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                            candidates.push_back(MatchCandidate{
                                .score = sc,
                                .parent_idx = p_idx,
                                .my_act = m_act,
                                .enemy_act = enemy_build_act,
                                .is_terminal = 0,
                                ._pad = 0
                            });
                        }
                    } else {
                        // 反撃・対応モード:
                        // 相手はこちらの攻撃に対し、最も痛い反撃・相殺 (Min手) を打ってくる
                        for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                            uint8_t worst_e_act = 0;
                            int32_t worst_sc = 2000000000;
                            for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                                PuyotanMatch esim = parent.match;
                                esim.setAction(my_id, getRLAction(m_act));
                                esim.setAction(enemy_id, getRLAction(e_act));
                                esim.stepUntilDecision();
                                int32_t sc2 = MatchBeamEvaluator::evaluate(esim, my_id, cfg);
                                if (sc2 < worst_sc) { worst_sc = sc2; worst_e_act = e_act; }
                            }

                            candidates.push_back(MatchCandidate{
                                .score = worst_sc,
                                .parent_idx = p_idx,
                                .my_act = m_act,
                                .enemy_act = worst_e_act,
                                .is_terminal = 0,
                                ._pad = 0
                            });
                        }
                    }

                } else if (my_turn) {
                    // 自分だけ手番: 22通り展開 (Max)
                    for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(my_id, getRLAction(m_act));
                        sim.stepUntilDecision();

                        int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
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
                    // 相手だけ手番:
                    const bool enemy_threatened = OpponentModel::isThreatened(parent.match, enemy_id);
                    uint8_t chosen_e_act = 0;
                    int32_t chosen_sc = 0;

                    if (!enemy_threatened) {
                        // 平時: 相手は最善の積み手を打つ
                        chosen_e_act = OpponentModel::selectBuildMove(parent.match, enemy_id, cfg);
                        PuyotanMatch sim = parent.match;
                        sim.setAction(enemy_id, getRLAction(chosen_e_act));
                        sim.stepUntilDecision();
                        chosen_sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                    } else {
                        // 被攻撃時: 相手は自分にとって最悪な手 (Min) を選ぶ
                        int32_t worst_sc = 2000000000;
                        for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                            PuyotanMatch sim = parent.match;
                            sim.setAction(enemy_id, getRLAction(e_act));
                            sim.stepUntilDecision();

                            int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
                            if (sc < worst_sc) {
                                worst_sc = sc;
                                chosen_e_act = e_act;
                            }
                        }
                        chosen_sc = worst_sc;
                    }

                    candidates.push_back(MatchCandidate{
                        .score = chosen_sc,
                        .parent_idx = p_idx,
                        .my_act = 255,
                        .enemy_act = chosen_e_act,
                        .is_terminal = 0,
                        ._pad = 0
                    });
                } else {
                    // 決定待ちなし（試合進行中）
                    PuyotanMatch sim = parent.match;
                    sim.stepUntilDecision();
                    int32_t sc = MatchBeamEvaluator::evaluate(sim, my_id, cfg);
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
            } else if (cand.my_act < kNumRLActions) {
                child.first_action = cand.my_act;
            } else {
                child.first_action = -1;
            }

            child.my_depth = parent.my_depth + (cand.my_act < kNumRLActions ? 1 : 0);

            // シミュレーションを実行して子状態を確定
            if (cand.my_act == 254) {
                child.match.setAction(my_id, Action{ActionType::Pass});
            } else if (cand.my_act < kNumRLActions) {
                child.match.setAction(my_id, getRLAction(cand.my_act));
            }
            if (cand.enemy_act == 254) {
                child.match.setAction(enemy_id, Action{ActionType::Pass});
            } else if (cand.enemy_act < kNumRLActions) {
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
