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
#include <puyotan/search/match_search.hpp>
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

// ────────────────────────────────────────────────────────────────────────────
// 盤面品質スコア（連結ボーナス・孤立ペナルティ・埋没ペナルティ）
// ────────────────────────────────────────────────────────────────────────────
inline int32_t boardQuality(const Board& board, const MatchBeamEvalWeights& w) noexcept {
    int32_t r = 0;

    __m128i all_has2 = _mm_setzero_si128();
    __m128i all_iso  = _mm_setzero_si128();

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.empty())
            continue;

        const __m128i bbm = bb.m128;
        const __m128i U = _mm_slli_epi64(bbm, 1);
        const __m128i D = _mm_srli_epi64(bbm, 1);
        const __m128i L = _mm_srli_si128(bbm, 2);
        const __m128i R = _mm_slli_si128(bbm, 2);

        const __m128i UD = _mm_or_si128(U, D);
        const __m128i LR = _mm_or_si128(L, R);

        // >= 2 same-color neighbors
        const __m128i has2 = _mm_and_si128(bbm,
            _mm_or_si128(_mm_or_si128(_mm_and_si128(U, D), _mm_and_si128(L, R)),
                         _mm_and_si128(UD, LR)));
        all_has2 = _mm_or_si128(all_has2, has2);

        // Isolated: no same-color neighbors
        const __m128i iso_m = _mm_andnot_si128(_mm_or_si128(UD, LR), bbm);
        all_iso = _mm_or_si128(all_iso, iso_m);
    }

    const BitBoard b_has2(all_has2);
    const BitBoard b_iso(all_iso);

    r += b_has2.popcount() * w.connectivity_bonus;
    r += b_iso.popcount()  * w.isolated_penalty;

    // Buried under ojama
    const BitBoard& oj = board.getBitboard(Cell::Ojama);
    if (!oj.empty()) {
        __m128i s_reg = oj.m128;
        s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 1));
        s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 2));
        s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 4));
        s_reg = _mm_or_si128(s_reg, _mm_srli_epi64(s_reg, 8));

        const __m128i all_colored = _mm_andnot_si128(oj.m128, board.getOccupied().m128);
        const BitBoard buried_bb(_mm_and_si128(all_colored, s_reg));
        r += buried_bb.popcount() * w.buried_penalty;
    }

    return r;
}

// ────────────────────────────────────────────────────────────────────────────
// 高さ危険ペナルティ：上段（危険閾値より上の列）の本数に比例
// ────────────────────────────────────────────────────────────────────────────
inline int32_t heightDangerPenalty(const Board& board, const MatchBeamEvalWeights& w) noexcept {
    // BitBoard では各列が epi16 の 1ワードに対応。occupied の各列の最上ビット位置が高さ。
    // 危険閾値 h_thresh を超えた列数をカウント。
    // occupied の epi16 各ワードをフラグとして使い、bit popcount で代替。
    // 簡易実装：packHeights の各列高さと比較
    const BitBoard& occ = board.getOccupied();
    if (occ.empty()) return 0;

    // 各列の高さを計算（epi16 の各ワードのビット数 = 列のぷよ数 = 高さ）
    const __m128i occ_m = occ.m128;
    // epi16 の各16bit ワードのbit countで列高さを得る（popcnt16相当をepi8で実行）
    // まず各バイトのbit count → 2バイトずつ加算
    const __m128i lut = _mm_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
    const __m128i lo_nibbles = _mm_and_si128(occ_m, _mm_set1_epi8(0x0F));
    const __m128i hi_nibbles = _mm_srli_epi16(_mm_and_si128(occ_m, _mm_set1_epi8(static_cast<char>(0xF0))), 4);
    const __m128i cnt8 = _mm_add_epi8(_mm_shuffle_epi8(lut, lo_nibbles),
                                       _mm_shuffle_epi8(lut, hi_nibbles));
    // 各epi16の高さ = 2バイトの和（madd with 1）
    const __m128i col_heights = _mm_maddubs_epi16(cnt8, _mm_set1_epi8(1));

    // height_danger_threshold を超えた列をカウント（比較はepi16符号付き）
    const __m128i thresh = _mm_set1_epi16(static_cast<int16_t>(w.height_danger_threshold));
    const __m128i over   = _mm_cmpgt_epi16(col_heights, thresh);
    // over の各-1ワードを1に変換してカウント
    const __m128i ones = _mm_and_si128(over, _mm_set1_epi16(1));
    // 水平加算
    const __m128i sum16 = _mm_add_epi16(ones, _mm_srli_si128(ones, 2));
    const __m128i sum32 = _mm_add_epi32(
        _mm_cvtepi16_epi32(sum16),
        _mm_cvtepi16_epi32(_mm_srli_si128(sum16, 8))
    );
    const __m128i sum64 = _mm_add_epi32(sum32, _mm_srli_si128(sum32, 4));
    const int danger_cols = _mm_cvtsi128_si32(sum64);

    return danger_cols * w.height_danger_penalty;
}

// ────────────────────────────────────────────────────────────────────────────
float sqrtPotential(int32_t raw_pot) noexcept {
    if (raw_pot <= 0) return 0.0f;
    return std::sqrt(static_cast<float>(raw_pot));
}

// ────────────────────────────────────────────────────────────────────────────
// 盤面静穏状態判定（連鎖中・アクション中・おじゃま保留・確定が一切ない状態）
// ────────────────────────────────────────────────────────────────────────────
inline bool isMatchQuiescent(const PuyotanMatch& m) noexcept {
    if (m.getStatus() != MatchStatus::Playing) return true;
    const auto& p0 = m.getPlayer(0);
    const auto& p1 = m.getPlayer(1);
    const bool p0_busy = (p0.chain_count > 0) ||
                         (p0.current_action.action.type != ActionType::None);
    const bool p1_busy = (p1.chain_count > 0) ||
                         (p1.current_action.action.type != ActionType::None);
    if (p0_busy || p1_busy) return false;
    if (p0.active_ojama > 0 || p1.active_ojama > 0 ||
        p0.non_active_ojama > 0 || p1.non_active_ojama > 0) {
        return false;
    }
    return true;
}

// ────────────────────────────────────────────────────────────────────────────
// 連鎖・おじゃま決着関数（置いた結果時点まで同期進行させて決着させる）
// ────────────────────────────────────────────────────────────────────────────
inline void resolveMatchToCleanState(PuyotanMatch& m, int max_steps = 80) noexcept {
    for (int s = 0; s < max_steps; ++s) {
        if (isMatchQuiescent(m)) break;
        int mask = m.getDecisionMask();
        if (mask & 1) {
            m.setAction(0, Action{ActionType::Pass});
        }
        if (mask & 2) {
            m.setAction(1, Action{ActionType::Pass});
        }
        m.stepUntilDecision();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// メイン評価関数（Match 全体を見た自分目線スコア）
// ────────────────────────────────────────────────────────────────────────────
inline int32_t evaluateMatchRaw(const PuyotanMatch& match, int my_id, const MatchBeamConfig& cfg) noexcept {
    const MatchStatus st = match.getStatus();
    const MatchBeamEvalWeights& w = cfg.eval_weights;
    const int enemy_id = 1 - my_id;
    const PuyotanPlayer& me    = match.getPlayer(my_id);
    const PuyotanPlayer& enemy = match.getPlayer(enemy_id);

    // ── 終局判定 ────────────────────────────────────────────────────────────
    if (st != MatchStatus::Playing) {
        const bool my_win  = (my_id == 0 && st == MatchStatus::WinP1)
                          || (my_id == 1 && st == MatchStatus::WinP2);
        const bool my_loss = (my_id == 0 && st == MatchStatus::WinP2)
                          || (my_id == 1 && st == MatchStatus::WinP1);
        if (my_win)  return  w.win_score  + (me.score - enemy.score);
        if (my_loss) return -w.win_score  + (me.score - enemy.score);
        return w.draw_score;
    }

    // ── ポテンシャルスコア（平方根正規化差分）──────────────────────────────────
    const uint32_t my_h     = packHeights(me.field);
    const uint32_t enemy_h  = packHeights(enemy.field);
    const int32_t my_raw_pot    = computeMaxPotentialScore(me.field, my_h);
    const int32_t enemy_raw_pot = computeMaxPotentialScore(enemy.field, enemy_h);
    // 対数圧縮を廃止し、平方根で正規化（指数的増加を恩和しつつ大連鎖の優位性を正しく反映）
    const int32_t my_sqrt_pot    = static_cast<int32_t>(sqrtPotential(my_raw_pot));
    const int32_t enemy_sqrt_pot = static_cast<int32_t>(sqrtPotential(enemy_raw_pot));
    const int32_t pot_diff = (my_sqrt_pot - enemy_sqrt_pot) * w.potential_score_scale;

    // ── 盤面形質差分（連結・孤立・埋没）──────────────────────────────────
    const int32_t my_quality    = boardQuality(me.field, w);
    const int32_t enemy_quality = boardQuality(enemy.field, w);
    const int32_t quality_diff  = my_quality - enemy_quality;

    // ── おじゃまペナルティ（自分のみ評価、相手へのおじゃま補正は0）────────────
    // ぷよたんは後出し優位のため、相手へのおじゃま送付による加点を0にする。
    // これにより平時における自発的な小連鎖の暴発を防ぎ、自分のおじゃま被弾回避・相殺に専念させる。
    const int32_t my_ao = static_cast<int32_t>(me.active_ojama);
    const int32_t my_ojama_active_pen  = w.active_ojama_coeff * my_ao * my_ao;
    const int32_t my_ojama_pending_pen = w.pending_ojama_penalty * static_cast<int32_t>(me.non_active_ojama);

    const int32_t ojama_diff = - my_ojama_active_pen - my_ojama_pending_pen;

    // ── 高さ危険ペナルティ（自分のみ）────────────────────────────────────
    const int32_t height_pen = heightDangerPenalty(me.field, w);

    // ── 平時発火ペナルティ ───────────────────────────────────────────────────────
    // 相手からのおじゃま攻撃がない平時に自分が連鎖を発火するのは暇打ちなので消極的に扱う
    int32_t reckless_pen = 0;
    if (w.reckless_fire_penalty_permille > 0) {
        const bool enemy_attacking = (enemy.chain_count > 0 ||
                                      enemy.active_ojama > 0 ||
                                      enemy.non_active_ojama > 0);
        const bool i_am_firing = (me.chain_count > 0);
        if (i_am_firing && !enemy_attacking) {
            // 平時に発火中: 発火後の相手のポテンシャル超過をペナルティ化
            reckless_pen = -w.reckless_fire_penalty_permille * my_sqrt_pot / 1000;
        }
    }

    // ── 累積スコア差（非終局時は対応モード時のみ考慮）────────────────────
    int32_t score_diff = 0;
    if (w.actual_score_weight > 0) {
        const bool enemy_attacking = (enemy.chain_count > 0 ||
                                      enemy.active_ojama > 0 ||
                                      enemy.non_active_ojama > 0);
        if (enemy_attacking) {
            // 対応モード（相手が攻撃中）では累積スコア差を考慮
            score_diff = (me.score - enemy.score) * w.actual_score_weight / 100;
        }
    }

    // ── 合計スコア ─────────────────────────────────────────────────────────────────
    return pot_diff + quality_diff + ojama_diff + height_pen + reckless_pen + score_diff;
}

// 連鎖中やおじゃま保留中がある場合は「置いた結果が決着するまで」同期進行させてから評価する
inline int32_t evaluateMatch(const PuyotanMatch& match, int my_id, const MatchBeamConfig& cfg) noexcept {
    if (isMatchQuiescent(match)) {
        return evaluateMatchRaw(match, my_id, cfg);
    }
    PuyotanMatch sim = match;
    resolveMatchToCleanState(sim, 80);
    return evaluateMatchRaw(sim, my_id, cfg);
}

// ────────────────────────────────────────────────────────────────────────────
// 平時ビルドアップモード判定：相手から攻撃が一切ない平和時は true
// ────────────────────────────────────────────────────────────────────────────
inline bool isBuildupMode(const PuyotanMatch& match, int my_id) noexcept {
    const auto& enemy = match.getPlayer(1 - my_id);
    return enemy.chain_count == 0
        && enemy.active_ojama == 0
        && enemy.non_active_ojama == 0;
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
                (ply >= min_turns && isMatchQuiescent(parent.match))) {
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

                        int32_t sc = evaluateMatch(sim, my_id, cfg);
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
                    int32_t sc = evaluateMatch(sim, my_id, cfg);
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

                        int32_t sc = evaluateMatch(sim, my_id, cfg);
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
            } else {
                // ────────────────────────────────────────────────────────────
                // 通常ターンフェーズ (ply < min_turns)
                // ────────────────────────────────────────────────────────────
                if (my_turn && enemy_turn) {
                    // 平時モード判定：相手から攻撃が一切ない場合は平時ビルドアップモード
                    const bool buildup = isBuildupMode(parent.match, my_id);

                    for (uint8_t m_act = 0; m_act < kNumRLActions; ++m_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(my_id, getRLAction(m_act));

                        if (buildup) {
                            // 平時ビルドアップモード: 相手はPass固定（静かに積んでいると仮定）
                            // 相手が次ターン発火してくるWorst-caseパニックを回避し、自分の連鎖構築に専念
                            sim.setAction(enemy_id, Action{ActionType::Pass});
                            sim.stepUntilDecision();
                        } else {
                            // 対応モード: 相手はMin手（自分にとって最悪な手）
                            uint8_t worst_e_act = 0;
                            int32_t worst_sc = 2000000000;
                            for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                                PuyotanMatch esim = parent.match;
                                esim.setAction(my_id, getRLAction(m_act));
                                esim.setAction(enemy_id, getRLAction(e_act));
                                esim.stepUntilDecision();
                                int32_t sc2 = evaluateMatch(esim, my_id, cfg);
                                if (sc2 < worst_sc) { worst_sc = sc2; worst_e_act = e_act; }
                            }
                            sim.setAction(enemy_id, getRLAction(worst_e_act));
                            sim.stepUntilDecision();
                        }

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

                } else if (my_turn) {
                    // 自分だけ手番: 22通り展開 (Max)
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
                    // 相手だけ手番: 相手は自分にとって最悪な手 (Min) を選ぶ
                    uint8_t worst_e_act = 0;
                    int32_t worst_sc = 2000000000;

                    for (uint8_t e_act = 0; e_act < kNumRLActions; ++e_act) {
                        PuyotanMatch sim = parent.match;
                        sim.setAction(enemy_id, getRLAction(e_act));
                        sim.stepUntilDecision();

                        int32_t sc = evaluateMatch(sim, my_id, cfg);
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
            if (cand.enemy_act < kNumRLActions) {
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
