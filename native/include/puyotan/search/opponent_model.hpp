#pragma once

#include <cstdint>

#include <puyotan/common/types.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/search/action_table.hpp>
#include <puyotan/search/beam_config.hpp>
#include <puyotan/search/match_evaluator.hpp>
#include <puyotan/search/potential_score.hpp>

namespace puyotan::search {

/**
 * @class OpponentModel
 * @brief Opponent move prediction and threat assessment for Match simulation beam search.
 *
 * 相手の行動予測（Opponent Modeling）をカプセル化するステートレスクラス。
 * 全てのメソッドは static inline / noexcept であり、オーバーヘッドなしで探索ループにインライン展開されます。
 */
class OpponentModel {
  public:
    // ────────────────────────────────────────────────────────────────────────
    // 相手がおじゃまの脅威に晒されているか判定（被攻撃・予告おじゃま・自連鎖中は true）
    // ────────────────────────────────────────────────────────────────────────
    static inline bool isThreatened(const PuyotanMatch& match, int enemy_id) noexcept {
        const int my_id = 1 - enemy_id;
        const auto& enemy = match.getPlayer(enemy_id);
        const auto& me    = match.getPlayer(my_id);
        return enemy.active_ojama > 0
            || enemy.non_active_ojama > 0
            || me.chain_count > 0;
    }

    // ────────────────────────────────────────────────────────────────────────
    // 相手が平時に連鎖を伸ばす最善手（ポテンシャル最大手）を1手選択
    // ────────────────────────────────────────────────────────────────────────
    static inline uint8_t selectBuildMove(const PuyotanMatch& match, int enemy_id, const MatchBeamConfig& cfg) noexcept {
        uint8_t best_act = 0;
        int32_t best_score = -2000000000;
        bool found_non_firing = false;

        for (uint8_t act = 0; act < kNumRLActions; ++act) {
            PuyotanMatch sim = match;
            sim.setAction(enemy_id, getRLAction(act));
            sim.stepUntilDecision();

            const auto& p = sim.getPlayer(enemy_id);
            const bool firing = (p.chain_count > 0);

            const uint32_t h = packHeights(p.field);
            const int32_t raw_pot = computeMaxPotentialScore(p.field, h);
            const int32_t sqrt_pot = static_cast<int32_t>(MatchBeamEvaluator::sqrtPotential(raw_pot));
            const int32_t sc = sqrt_pot * cfg.eval_weights.potential_score_scale 
                             + MatchBeamEvaluator::boardQuality(p.field, cfg.eval_weights);

            // 平時なので非発火手を優先（無駄な暴発を防ぎビルドアップに専念）
            if (!firing) {
                if (!found_non_firing || sc > best_score) {
                    best_score = sc;
                    best_act = act;
                    found_non_firing = true;
                }
            } else if (!found_non_firing) {
                if (sc > best_score) {
                    best_score = sc;
                    best_act = act;
                }
            }
        }
        return best_act;
    }
};

} // namespace puyotan::search
