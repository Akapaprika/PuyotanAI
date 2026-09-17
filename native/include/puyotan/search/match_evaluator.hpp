#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/potential_score.hpp>

namespace puyotan::search {

/**
 * @class MatchBeamEvaluator
 * @brief Stateless board and match evaluator for use in 1v1 Match simulation beam search.
 *
 * 全てのメソッドは static inline / noexcept であり、状態を持ちません。
 * コンパイラによるインライン展開により、呼び出しオーバーヘッドはゼロです。
 */
class MatchBeamEvaluator {
  public:
    // ────────────────────────────────────────────────────────────────────────
    // ポテンシャルの平方根正規化
    // ────────────────────────────────────────────────────────────────────────
    static inline float sqrtPotential(int32_t raw_pot) noexcept {
        if (raw_pot <= 0) return 0.0f;
        return std::sqrt(static_cast<float>(raw_pot));
    }

    // ────────────────────────────────────────────────────────────────────────
    // 盤面品質スコア（連結ボーナス・孤立ペナルティ・埋没ペナルティ）
    // ────────────────────────────────────────────────────────────────────────
    static inline int32_t boardQuality(const Board& board, const MatchBeamEvalWeights& w) noexcept {
        // 重みがすべて0なら評価処理を完全にスキップ
        if (w.connectivity_bonus == 0 && w.isolated_penalty == 0 && w.buried_penalty == 0) {
            return 0;
        }

        int32_t r = 0;

        // 連結・孤立判定（どちらかの重みが非ゼロの時のみSIMD走査を実行）
        if (w.connectivity_bonus != 0 || w.isolated_penalty != 0) {
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

            if (w.connectivity_bonus != 0) r += b_has2.popcount() * w.connectivity_bonus;
            if (w.isolated_penalty != 0)   r += b_iso.popcount()  * w.isolated_penalty;
        }

        // Buried under ojama（埋没ペナルティが有効な時のみ実行）
        if (w.buried_penalty != 0) {
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
        }

        return r;
    }

    // ────────────────────────────────────────────────────────────────────────
    // 高さ危険ペナルティ：上段（危険閾値より上の列）の本数に比例（SIMD高速化版）
    // ────────────────────────────────────────────────────────────────────────
    static inline int32_t heightDangerPenalty(const Board& board, const MatchBeamEvalWeights& w) noexcept {
        if (w.height_danger_penalty == 0) {
            return 0;
        }

        const BitBoard& occ = board.getOccupied();
        if (occ.empty()) return 0;

        const __m128i occ_m = occ.m128;
        const __m128i lut = _mm_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
        const __m128i lo_nibbles = _mm_and_si128(occ_m, _mm_set1_epi8(0x0F));
        const __m128i hi_nibbles = _mm_srli_epi16(_mm_and_si128(occ_m, _mm_set1_epi8(static_cast<char>(0xF0))), 4);
        const __m128i cnt8 = _mm_add_epi8(_mm_shuffle_epi8(lut, lo_nibbles),
                                           _mm_shuffle_epi8(lut, hi_nibbles));
        const __m128i col_heights = _mm_maddubs_epi16(cnt8, _mm_set1_epi8(1));

        const __m128i thresh = _mm_set1_epi16(static_cast<int16_t>(w.height_danger_threshold));
        const __m128i over   = _mm_cmpgt_epi16(col_heights, thresh);
        const __m128i ones   = _mm_and_si128(over, _mm_set1_epi16(1));
        const __m128i sum16  = _mm_add_epi16(ones, _mm_srli_si128(ones, 2));
        const __m128i sum32  = _mm_add_epi32(
            _mm_cvtepi16_epi32(sum16),
            _mm_cvtepi16_epi32(_mm_srli_si128(sum16, 8))
        );
        const __m128i sum64  = _mm_add_epi32(sum32, _mm_srli_si128(sum32, 4));
        const int danger_cols = _mm_cvtsi128_si32(sum64);

        return danger_cols * w.height_danger_penalty;
    }

    // ────────────────────────────────────────────────────────────────────────
    // 静穏状態判定（両者ともに連鎖・アクション・保留おじゃまがない状態）
    // ────────────────────────────────────────────────────────────────────────
    static inline bool isMatchQuiescent(const PuyotanMatch& m) noexcept {
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

    // ────────────────────────────────────────────────────────────────────────
    // 連鎖・おじゃま決着関数（置いた結果時点まで同期進行させて決着させる）
    // ────────────────────────────────────────────────────────────────────────
    static inline void resolveMatchToCleanState(PuyotanMatch& m, int max_steps = 80) noexcept {
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

    // ────────────────────────────────────────────────────────────────────────
    // 局面評価（Raw: 現在の盤面・Match状態に対する評価スコア）
    // ────────────────────────────────────────────────────────────────────────
    static inline int32_t evaluateRaw(const PuyotanMatch& match, int my_id, const MatchBeamConfig& cfg) noexcept {
        const MatchStatus st = match.getStatus();
        const MatchBeamEvalWeights& w = cfg.eval_weights;
        const int enemy_id = 1 - my_id;
        const PuyotanPlayer& me    = match.getPlayer(my_id);
        const PuyotanPlayer& enemy = match.getPlayer(enemy_id);

        // ── 終局判定 ────────────────────────────────────────────────────────
        if (st != MatchStatus::Playing) {
            const bool my_win  = (my_id == 0 && st == MatchStatus::WinP1)
                              || (my_id == 1 && st == MatchStatus::WinP2);
            const bool my_loss = (my_id == 0 && st == MatchStatus::WinP2)
                              || (my_id == 1 && st == MatchStatus::WinP1);
            if (my_win)  return  w.win_score;
            if (my_loss) return -500000;
            return w.draw_score;
        }

        // ── ポテンシャル ＋ 連鎖バリエーション（工夫案A：累積偏差和）────────────
        // 2ぷよ落とし（576通り）により、大連鎖の保持力と幅広い発火点（小連鎖対応手）を同時に評価
        const float div_w = w.diversity_weight_permille / 1000.0f;
        const auto port = computeChainPortfolio(me.field, div_w);
        const int32_t pot_score = static_cast<int32_t>(port.total_score * w.potential_score_scale);

        // ── 自分の盤面形質（連結・孤立・埋没）────────────────────────────
        const int32_t quality_score = boardQuality(me.field, w);

        // ── おじゃまペナルティ（自分のみ評価、相手へのおじゃま補正は0）────────
        // ぷよたんは後出し優位のため、相手へのおじゃま送付による加点を0にする。
        const int32_t my_ao = static_cast<int32_t>(me.active_ojama);
        const int32_t my_ojama_active_pen  = w.active_ojama_coeff * my_ao * my_ao;
        const int32_t my_ojama_pending_pen = w.pending_ojama_penalty * static_cast<int32_t>(me.non_active_ojama);
        const int32_t ojama_diff = - my_ojama_active_pen - my_ojama_pending_pen;

        // ── 高さ危険ペナルティ（自分のみ）────────────────────────────────
        const int32_t height_pen = heightDangerPenalty(me.field, w);

        // ── 相手の攻撃状態判定 ───────────────────────────────────────────
        // 相手が連鎖中、または自分におじゃまが予告（落下確定・保留）されている状態を攻撃中と判定
        const bool enemy_attacking = (enemy.chain_count > 0 ||
                                      me.active_ojama > 0 ||
                                      me.non_active_ojama > 0);

        // ── 平時発火ペナルティ ───────────────────────────────────────────
        // 相手からのおじゃま攻撃がない平時に自分が連鎖を発火するのは暇打ちなので消極的に扱う
        int32_t reckless_pen = 0;
        if (w.reckless_fire_penalty_permille > 0) {
            const bool i_am_firing = (me.chain_count > 0);
            if (i_am_firing && !enemy_attacking) {
                // 平時に発火中: 発火後の相手のポテンシャル超過をペナルティ化
                reckless_pen = -w.reckless_fire_penalty_permille * static_cast<int32_t>(port.max_sqrt) / 1000;
            }
        }

        // ── 累積スコア差（非終局時は対応モード時のみ考慮）────────────────
        int32_t score_diff = 0;
        if (w.actual_score_weight > 0) {
            if (enemy_attacking) {
                // 対応モード（相手が攻撃中）では累積スコア差を考慮
                score_diff = (me.score - enemy.score) * w.actual_score_weight / 100;
            }
        }

        // ── 合計スコア ─────────────────────────────────────────────────
        return pot_score + quality_score + ojama_diff + height_pen + reckless_pen + score_diff;
    }

    // ────────────────────────────────────────────────────────────────────────
    // 評価メインエントリ（連鎖中等なら結果が決着するまで同期進行させてから評価）
    // ────────────────────────────────────────────────────────────────────────
    static inline int32_t evaluate(const PuyotanMatch& match, int my_id, const MatchBeamConfig& cfg) noexcept {
        if (isMatchQuiescent(match)) {
            return evaluateRaw(match, my_id, cfg);
        }
        PuyotanMatch sim = match;
        resolveMatchToCleanState(sim, 80);
        return evaluateRaw(sim, my_id, cfg);
    }
};

} // namespace puyotan::search
