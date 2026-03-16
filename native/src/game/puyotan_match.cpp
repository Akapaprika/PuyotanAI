#include <puyotan/game/puyotan_match.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>
#include <cmath>

namespace puyotan {

void PuyotanPlayer::fallOjama(int num, uint32_t& seed) {
    auto nextInt = [&](int max) {
        int32_t signed_y = static_cast<int32_t>(seed);
        signed_y ^= (signed_y << 13);
        signed_y ^= (signed_y >> 17);
        signed_y ^= (signed_y << 15);
        seed = static_cast<uint32_t>(signed_y);
        int r = std::abs(signed_y);
        return r % max;
    };

    while (num > 0) {
        if (num >= 6) {
            for (int x = 0; x < 6; ++x) {
                field.set(x, config::Board::kSpawnRow, Cell::Ojama);
            }
            Gravity::execute(field);
            num -= 6;
        } else {
            bool memo[6] = {false};
            for (int i = 0; i < num; ++i) {
                int pos = nextInt(6 - i);
                int cnt = 0;
                for (int j = 0; j < 6; ++j) {
                    if (!memo[j]) {
                        if (cnt++ == pos) {
                            memo[j] = true;
                            break;
                        }
                    }
                }
            }
            for (int x = 0; x < 6; ++x) {
                if (memo[x]) {
                    field.set(x, config::Board::kSpawnRow, Cell::Ojama);
                }
            }
            Gravity::execute(field);
            num = 0;
        }
    }
}

PuyotanMatch::PuyotanMatch(uint32_t seed) : seed_(seed), tsumo_(seed) {}

void PuyotanMatch::start() {
    if (frame_ == 0) {
        ++frame_;
        status_ = MatchStatus::PLAYING;
    }
}

std::string PuyotanMatch::getStatusText() const {
    switch (status_) {
        case MatchStatus::READY:   return "待機中";
        case MatchStatus::PLAYING: return "対戦中";
        case MatchStatus::WIN_P1:  return "Player 1 の勝利！";
        case MatchStatus::WIN_P2:  return "Player 2 の勝利！";
        case MatchStatus::DRAW:    return "引き分け…";
        default:                   return "不明";
    }
}

bool PuyotanMatch::setAction(int id, Action action) {
    if (frame_ <= 0) return false;
    if (players_[id].action_histories.find(frame_) == players_[id].action_histories.end()) {
        switch (action.type) {
            case ActionType::PASS:
                players_[id].action_histories[frame_] = {action, 0};
                return true;
            case ActionType::PUT:
                players_[id].action_histories[frame_] = {action, 1};
                return true;
            default:
                // JS throws an error, we return false
                return false;
        }
    }
    return false;
}

bool PuyotanMatch::canStepNextFrame() const {
    if (frame_ <= 0) return false;
    for (int id = 0; id < 2; ++id) {
        if (players_[id].action_histories.count(frame_) == 0) return false;
    }
    return true;
}

void PuyotanMatch::stepNextFrame() {
    if (!canStepNextFrame()) return;

    // 2. 行動選択・予約
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto current_it = p.action_histories.find(frame_);
        if (current_it != p.action_histories.end() && current_it->second.remaining_frame > 0) {
            p.action_histories[frame_ + 1] = {current_it->second.action, current_it->second.remaining_frame - 1};
        }
    }

    // 3. 行動実行
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto current_it = p.action_histories.find(frame_);
        if (current_it != p.action_histories.end() && current_it->second.remaining_frame == 0) {
            const auto& action = current_it->second.action;
            switch (action.type) {
                case ActionType::PASS:
                    break;
                case ActionType::PUT: {
                    PuyoPiece tumo = tsumo_.get(p.active_next_pos);
                    int x = action.x;
                    Rotation rotation = action.rotation;
                    
                    int sub_x = x;
                    int sub_y = config::Board::kSpawnRow;
                    switch (rotation) {
                        case Rotation::Up:    sub_y += 1; break;
                        case Rotation::Right: sub_x += 1; break;
                        case Rotation::Down:  sub_y -= 1; break;
                        case Rotation::Left:  sub_x -= 1; break;
                    }

                    // 1. Calculate Soft Drop Bonus (before placement)
                    int d1 = p.field.getDropDistance(x, config::Board::kSpawnRow);
                    int drop_dist = d1;
                    if (sub_x >= 0 && sub_x < config::Board::kWidth) {
                        int d2 = p.field.getDropDistance(sub_x, sub_y);
                        drop_dist = std::min(d1, d2);
                    }
                    p.score += std::max(0, drop_dist) * config::Score::kSoftDropBonusPerGrid;

                    // 2. Placement
                    switch (rotation) {
                        case Rotation::Up:    
                            p.field.set(x, config::Board::kSpawnRow, tumo.axis);
                            p.field.set(x, config::Board::kSpawnRow + 1, tumo.sub);
                            break;
                        case Rotation::Right: 
                            p.field.set(x, config::Board::kSpawnRow, tumo.axis);
                            if (x + 1 < config::Board::kWidth) p.field.set(x + 1, config::Board::kSpawnRow, tumo.sub);
                            break;
                        case Rotation::Down:  
                            p.field.set(x, config::Board::kSpawnRow, tumo.sub);
                            p.field.set(x, config::Board::kSpawnRow + 1, tumo.axis);
                            break;
                        case Rotation::Left:  
                            p.field.set(x, config::Board::kSpawnRow, tumo.axis);
                            if (x - 1 >= 0) p.field.set(x - 1, config::Board::kSpawnRow, tumo.sub);
                            break;
                    }

                    Gravity::execute(p.field);

                    if (Chain::canFire(p.field)) {
                        p.chain_count = 0;
                        p.action_histories[frame_ + 1] = {Action{ActionType::CHAIN}, 1};
                    }
                    break;
                }
                case ActionType::CHAIN: {
                    ErasureData info = Chain::execute(p.field);
                    p.chain_count++;
                    p.score += Scorer::calculateStepScore(info, p.chain_count);
                    
                    int ojama = (p.score - p.used_score) / 70;
                    p.used_score += ojama * 70;
                    
                    if (p.non_active_ojama > 0) {
                        int used = std::min(ojama, p.non_active_ojama);
                        p.non_active_ojama -= used;
                        ojama -= used;
                    }
                    if (p.active_ojama > 0) {
                        int used = std::min(ojama, p.active_ojama);
                        p.active_ojama -= used;
                        ojama -= used;
                    }
                    if (ojama > 0) {
                        sendOjama(id, ojama);
                    }
                    
                    // All Clear check (simplified)
                    if (p.field.getOccupied().empty()) {
                        p.score += 2100;
                    }

                    if (Gravity::canFall(p.field)) {
                        p.action_histories[frame_ + 1] = {Action{ActionType::CHAIN_FALL}, 0};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::CHAIN_FALL: {
                    Gravity::execute(p.field);
                    if (Chain::canFire(p.field)) {
                        p.action_histories[frame_ + 1] = {Action{ActionType::CHAIN}, 1};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::OJAMA: {
                    int fall_num = std::min(p.active_ojama, 30);
                    p.active_ojama -= fall_num;
                    p.fallOjama(fall_num, seed_);
                    break;
                }
                default:
                    break;
            }
        }
    }

    // 4. 窒息判定
    int alive_count = 0;
    int alive_player_id = 0;
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        // Check death cell (3, 12) -> x=2, y=11 (0-indexed)
        if (p.action_histories.count(frame_ + 1) == 0 && p.field.get(2, 11) != Cell::Empty) {
            // Death
        } else {
            alive_count++;
            alive_player_id = id;
        }
    }

    if (alive_count == 0) {
        status_ = MatchStatus::DRAW;
    } else if (alive_count == 1) {
        status_ = (alive_player_id == 0) ? MatchStatus::WIN_P1 : MatchStatus::WIN_P2;
    }

    // 5. おじゃま処理
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        bool is_currently_ojama = false;
        if (p.action_histories.count(frame_) > 0) {
            is_currently_ojama = (p.action_histories[frame_].action.type == ActionType::OJAMA);
        }
        if (!is_currently_ojama && p.action_histories.count(frame_ + 1) == 0 && p.active_ojama > 0) {
            p.action_histories[frame_ + 1] = {Action{ActionType::OJAMA}, 0};
        }
    }

    // 0. フレーム遷移
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        bool current_is_pass = false;
        if (p.action_histories.count(frame_) > 0) {
            current_is_pass = (p.action_histories[frame_].action.type == ActionType::PASS);
        }
        if (!current_is_pass && p.action_histories.count(frame_ + 1) == 0) {
            p.active_next_pos++;
        }
    }

    ++frame_;
}

void PuyotanMatch::sendOjama(int sender_id, int ojama) {
    int target_id = 1 - sender_id;
    players_[target_id].non_active_ojama += ojama;
}

void PuyotanMatch::activateOjama(int sender_id) {
    int target_id = 1 - sender_id;
    auto& p = players_[target_id];
    p.active_ojama += p.non_active_ojama;
    p.non_active_ojama = 0;
}

} // namespace puyotan
