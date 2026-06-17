#include <algorithm>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/scorer.hpp>
namespace puyotan {
static constexpr auto kOjamaColumnLut = []() consteval {
    std::array<std::array<uint8_t, 6>, 64> arr{};
    for (int free_mask = 0; free_mask < 64; ++free_mask) {
        for (int pos = 0; pos < 6; ++pos) {
            int temp = free_mask;
            int target_bit = 0;
            int count = 0;
            for (int b = 0; b < 6; ++b) {
                if ((temp >> b) & 1) {
                    if (count == pos) {
                        target_bit = 1 << b;
                        break;
                    }
                    ++count;
                }
            }
            arr[free_mask][pos] = static_cast<uint8_t>(target_bit);
        }
    }
    return arr;
}();

void PuyotanPlayer::fallOjama(int num, uint32_t& seed) noexcept {
    constexpr int width = config::Board::kWidth;
    while (num > 0) {
        uint32_t mask;
        int drop_num;

        if (num >= width) {
            mask = 0x3F;
            drop_num = width;
        } else {
            mask = 0;
            for (int i = 0; i < num; ++i) {
                const int pos = PuyotanMatch::nextInt(seed, width - i);
                uint32_t free = ~mask & 0x3F;
                mask |= kOjamaColumnLut[free][pos];
            }
            drop_num = num;
        }

        field.setRowMask(config::Board::kSpawnRow, Cell::Ojama, mask);
        Gravity::execute(field);
        num -= drop_num;
    }
}

int PuyotanMatch::getDecisionMask() const noexcept {
    if (status_ != MatchStatus::Playing) [[unlikely]]
        return 0;

    int p0_none = static_cast<int>(players_[0].current_action.action.type ==
                                   ActionType::None);
    int p1_none = static_cast<int>(players_[1].current_action.action.type ==
                                   ActionType::None);
    return p0_none | (p1_none << 1);
}

PuyotanMatch::PuyotanMatch(uint32_t seed) noexcept : tsumo_(seed) {
    assert(seed != 0u);
    seed_ = tsumo_.getSeed();
}

void PuyotanMatch::start() noexcept {
    assert(status_ == MatchStatus::Ready &&
           "start() should only be called once when match is ready");
    status_ = MatchStatus::Playing;
}

void PuyotanMatch::stepNextFrame() noexcept {
    if (!canStepNextFrame())
        return;

    // 開始時のアクションタイプをローカルスタック（レジスタ）に保持しておく
    std::array<ActionType, config::Rule::kNumPlayers> prev_types;
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        prev_types[id] = players_[id].current_action.action.type;
    }

    // 1. Execute or reserve actions
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];

        // 残りフレームがある場合はデクリメントするだけ
        if (p.current_action.remaining_frame > 0) {
            p.current_action.remaining_frame--;
        } else {
            const auto& action = p.current_action.action;
            switch (action.type) {
                case ActionType::Pass:
                    p.current_action = {}; // 終了したのでNoneクリア
                    break;
                case ActionType::Put: {
                    const PuyoPiece tumo = tsumo_.get(p.active_next_pos);
                    const int r = static_cast<int>(action.rotation);
                    const int x_axis = action.x;
                    const int x_sub = x_axis + kSubDx[r];
                    const int h_axis = p.field.getColumnHeight(x_axis);
                    const int h_sub = p.field.getColumnHeight(x_sub);
                    p.score += std::max(0, config::Board::kSpawnRow -
                                               std::max(h_axis, h_sub));

                    const int y_axis = h_axis + kAxisDy[r];
                    const int y_sub = h_sub + kSubDySimple[r];
                    p.field.dropNewPiece(x_axis, y_axis, tumo.axis);
                    p.field.dropNewPiece(x_sub, y_sub, tumo.sub);

                    const uint32_t dirty_colors = tumo.dirty_flag;
                    Chain::scanGroups(p.field, pending_erasure_[id],
                                      dirty_colors);

                    if (pending_erasure_[id].num_erased > 0) {
                        p.current_action = {Action{ActionType::Chain}, 1};
                    } else {
                        p.current_action = {}; // 連鎖なし、終了クリア
                    }
                    break;
                }
                case ActionType::Chain: {
                    Chain::applyErasure(p.field, pending_erasure_[id]);
                    const ErasureData& info = pending_erasure_[id];
                    ++p.chain_count;
                    int step_score =
                        Scorer::calculateStepScore(info, p.chain_count);
                    p.score += step_score;
                    int ojama =
                        (p.score - p.used_score) / config::Score::kTargetScore;
                    p.used_score += ojama * config::Score::kTargetScore;

                    int used_non =
                        std::min(ojama, static_cast<int>(p.non_active_ojama));
                    p.non_active_ojama -= static_cast<uint16_t>(used_non);
                    ojama -= used_non;
                    int used_active =
                        std::min(ojama, static_cast<int>(p.active_ojama));
                    p.active_ojama -= static_cast<uint16_t>(used_active);
                    ojama -= used_active;
                    sendOjama(id, ojama);

                    bool field_empty = p.field.getOccupied().empty();
                    p.score += static_cast<int>(field_empty) *
                               config::Score::kAllClearBonus;

                    if (Gravity::canFall(p.field)) {
                        p.current_action = {Action{ActionType::ChainFall}, 0};
                    } else {
                        p.chain_count = 0;
                        activateOjama(id);
                        p.current_action = {}; // 終了クリア
                    }
                    break;
                }
                case ActionType::ChainFall: {
                    uint32_t dirty_colors = Gravity::execute(p.field);
                    Chain::scanGroups(p.field, pending_erasure_[id],
                                      dirty_colors);
                    if (pending_erasure_[id].num_erased > 0) {
                        p.current_action = {Action{ActionType::Chain}, 1};
                    } else {
                        p.chain_count = 0;
                        activateOjama(id);
                        p.current_action = {}; // 終了クリア
                    }
                    break;
                }
                case ActionType::Ojama: {
                    int fall_num = std::min(static_cast<int>(p.active_ojama),
                                            config::Rule::kMaxOjamaPerFall);
                    p.active_ojama -= static_cast<uint16_t>(fall_num);
                    p.fallOjama(fall_num, seed_);
                    p.current_action = {}; // 終了クリア
                    break;
                }
                default:
                    break;
            }
        }
    }

    // 3. Death check
    uint32_t alive_mask = 0;
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];
        // 【変更】：current_action を直接チェックします
        bool is_alive = (p.current_action.action.type != ActionType::None) |
                        !p.field.isOccupied(config::Rule::kDeathCol,
                                            config::Rule::kDeathRow);
        alive_mask |= (is_alive << id);
    }

    static_assert(config::Rule::kNumPlayers == 2,
                  "Match status mapping explicitly assumes 2 players");
    if (alive_mask != 3) {
        static constexpr MatchStatus kNextStatus[] = {
            MatchStatus::Draw, MatchStatus::WinP1, MatchStatus::WinP2,
            MatchStatus::Playing};
        status_ = kNextStatus[alive_mask];
    }

    // 4 & 5. Post-turn processing
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];
        if (p.current_action.action.type == ActionType::None) {
            // 4. Ojama (garbage) processing
            // 開始時のアクションタイプである prev_types を用いて判定します
            if (p.active_ojama > 0 && prev_types[id] != ActionType::Ojama) {
                p.current_action = {Action{ActionType::Ojama}, 0};
            }
            // 5. Tsumo and frame transition
            else if (prev_types[id] != ActionType::Pass) {
                if (p.active_next_pos == 999) [[unlikely]] {
                    p.active_next_pos = 0;
                } else {
                    ++(p.active_next_pos);
                }
            }
        }
    }
    ++frame_;
}

void PuyotanMatch::sendOjama(int sender_id, int ojama) noexcept {
    int target_id = 1 - sender_id;
    players_[target_id].non_active_ojama += ojama;
}

void PuyotanMatch::activateOjama(int finishing_player_id) noexcept {
    int target_id = 1 - finishing_player_id;
    auto& p = players_[target_id];
    p.active_ojama += p.non_active_ojama;
    p.non_active_ojama = 0;
}

int PuyotanMatch::stepUntilDecision() noexcept {
    while (status_ == MatchStatus::Playing) {
        int mask = getDecisionMask();
        if (mask != 0)
            return mask;

        stepNextFrame();
    }
    return 0;
}

int64_t PuyotanMatch::runBatch(int num_games, uint32_t seed) noexcept {
    int64_t total_frames = 0;
    // 6 at col 5, 6 at 4, 6 at 3, etc.
    const int move_plan[] = {5, 5, 5, 5, 5, 5, 4, 4, 4,
                             4, 4, 4, 3, 3, 3, 3, 3, 3};
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);

    for (int i = 0; i < num_games; ++i) {
        PuyotanMatch match(seed + static_cast<uint32_t>(i));
        match.start();
        int p_move = 0;

        while (match.getStatus() == MatchStatus::Playing) {
            bool action_set = false;

            if (match.players_[0].current_action.action.type ==
                ActionType::None) {
                int col = (p_move < num_moves) ? move_plan[p_move] : 2;
                Action act{ActionType::Put, static_cast<int8_t>(col),
                           Rotation::Up};

                if (match.setAction(0, act) && match.setAction(1, act)) {
                    ++p_move;
                    action_set = true;
                }
            }

            if (match.canStepNextFrame()) {
                match.stepNextFrame();
                ++total_frames;
            } else if (!action_set) {
                // デッドロック防止
                break;
            }

            // Failsafe
            if (match.frame_ > 3000)
                break;
        }
    }
    return total_frames;
}

int PuyotanMatch::nextInt(uint32_t& seed, int max) noexcept {
    assert(seed != 0u);
    seed ^= (seed << 13);
    seed ^= static_cast<uint32_t>(static_cast<int32_t>(seed) >> 17);
    seed ^= (seed << 15);
    return static_cast<int>(seed % static_cast<uint32_t>(max));
}
} // namespace puyotan
