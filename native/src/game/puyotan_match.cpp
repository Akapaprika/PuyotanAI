#include <puyotan/game/puyotan_match.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>

namespace puyotan {

void PuyotanPlayer::fallOjama(int num, int32_t& seed) {
    constexpr int width = config::Board::kWidth;
    while (num > 0) {
        if (num >= width) {
            for (int x = 0; x < width; ++x) {
                field.set(x, config::Board::kSpawnRow, Cell::Ojama);
            }
            Gravity::execute(field);
            num -= width;
        } else {
            uint8_t mask = 0;
            for (int i = 0; i < num; ++i) {
                int pos = PuyotanMatch::nextInt(seed, width - i);
                int cnt = 0;
                for (int x = 0; x < width; ++x) {
                    if (!(mask & (1 << x))) {
                        if (cnt++ == pos) {
                            mask |= (1 << x);
                            break;
                        }
                    }
                }
            }
            for (int x = 0; x < width; ++x) {
                if (mask & (1 << x)) {
                    field.set(x, config::Board::kSpawnRow, Cell::Ojama);
                }
            }
            Gravity::execute(field);
            num = 0;
        }
    }
}

PuyotanMatch::PuyotanMatch(int32_t seed) : seed_(seed), tsumo_(seed) {}

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
    auto& history = players_[id].action_histories[frame_ & 255];
    if (history.action.type == ActionType::NONE) {
        switch (action.type) {
            case ActionType::PASS:
                history = {action, 0};
                return true;
            case ActionType::PUT:
                history = {action, 1};
                return true;
            default:
                return false;
        }
    }
    return false;
}

bool PuyotanMatch::canStepNextFrame() const {
    if (frame_ <= 0) return false;
    for (int id = 0; id < 2; ++id) {
        if (players_[id].action_histories[frame_ & 255].action.type == ActionType::NONE) return false;
    }
    return true;
}

void PuyotanMatch::stepNextFrame() {
    if (!canStepNextFrame()) return;

    // 2. 行動選択・予約
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto& current = p.action_histories[frame_ & 255];
        if (current.action.type != ActionType::NONE && current.remaining_frame > 0) {
            p.action_histories[(frame_ + 1) & 255] = {current.action, static_cast<uint8_t>(current.remaining_frame - 1)};
        }
    }

    // 3. 行動実行
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto& current = p.action_histories[frame_ & 255];
        if (current.action.type != ActionType::NONE && current.remaining_frame == 0) {
            const auto& action = current.action;
            switch (action.type) {
                case ActionType::PASS:
                    break;
                case ActionType::PUT: {
                    PuyoPiece tumo = tsumo_.get(p.active_next_pos);
                    const int x = action.x;
                    const int r = static_cast<int>(action.rotation);
                    assert(x >= 0 && x < config::Board::kWidth);
                    assert(r >= 0 && r < 4);


                    const int h_axis = p.field.getColumnHeight(x);
                    const int sub_dx = kSubDx[r];
                    const int sub_x  = x + sub_dx;
                    assert(sub_x >= 0 && sub_x < config::Board::kWidth);

                    const int h_sub = p.field.getColumnHeight(sub_x);

                    const int final_y_axis = h_axis + kAxisDy[r];
                    const int final_y_sub  = h_sub  + kSubDy_Simple[r];

                    const int drop_dist   = config::Board::kSpawnRow - std::max(h_axis, h_sub);
                    p.score += drop_dist * config::Score::kSoftDropBonusPerGrid;

                    p.field.dropNewPiece(x, final_y_axis, tumo.axis);
                    p.field.dropNewPiece(sub_x, final_y_sub, tumo.sub);

                    uint8_t dirty_colors = (1 << static_cast<int>(tumo.axis)) | (1 << static_cast<int>(tumo.sub));
                    if (Chain::canFire(p.field, dirty_colors)) {
                        p.chain_count = 0;
                        p.action_histories[(frame_ + 1) & 255] = {Action{ActionType::CHAIN}, 1};
                    }
                    break;
                }
                case ActionType::CHAIN: {
                    ErasureData info = Chain::execute(p.field);
                    ++p.chain_count;
                    p.score += Scorer::calculateStepScore(info, p.chain_count);
                    
                    int ojama = (p.score - p.used_score) / config::Score::kTargetScore;
                    p.used_score += ojama * config::Score::kTargetScore;
                    
                    if (p.non_active_ojama > 0) {
                        int used = std::min(ojama, static_cast<int>(p.non_active_ojama));
                        p.non_active_ojama -= static_cast<uint16_t>(used);
                        ojama -= used;
                    }
                    if (p.active_ojama > 0) {
                        int used = std::min(ojama, static_cast<int>(p.active_ojama));
                        p.active_ojama -= static_cast<uint16_t>(used);
                        ojama -= used;
                    }
                    if (ojama > 0) {
                        sendOjama(id, ojama);
                    }
                    
                    // All Clear check
                    if (p.field.getOccupied().empty()) {
                        p.score += config::Score::kAllClearBonus;
                    }

                    if (Gravity::canFall(p.field)) {
                        p.action_histories[(frame_ + 1) & 255] = {Action{ActionType::CHAIN_FALL}, 0};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::CHAIN_FALL: {
                    uint8_t dirty_colors = Gravity::execute(p.field);
                    if (Chain::canFire(p.field, dirty_colors)) {
                        p.chain_count = 0; // wait, puyotan_match handles chain_count in CHAIN action. 
                        // Actually, puyotan_match didn't reset it here before. Let's keep original logic.
                        p.action_histories[(frame_ + 1) & 255] = {Action{ActionType::CHAIN}, 1};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::OJAMA: {
                    int fall_num = std::min(static_cast<int>(p.active_ojama), config::Rule::kMaxOjamaPerFall);
                    p.active_ojama -= static_cast<uint16_t>(fall_num);
                    p.fallOjama(fall_num, seed_);
                    break;
                }
                default:
                    break;
            }
        }
    }

    // 4. 窒息判定 (Branchless Status Map)
    uint32_t alive_mask = 0;
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        bool is_alive = p.action_histories[(frame_ + 1) & 255].action.type != ActionType::NONE || 
                        p.field.get(config::Rule::kDeathCol, config::Rule::kDeathRow) == Cell::Empty;
        alive_mask |= (is_alive << id);
    }

    if (alive_mask != 3) { // 少なくとも一人が死亡
        static constexpr MatchStatus kNextStatus[] = {
            MatchStatus::DRAW,   // 00: 両者死亡
            MatchStatus::WIN_P1, // 01: P1生存・P2死亡
            MatchStatus::WIN_P2, // 10: P1死亡・P2生存
            MatchStatus::PLAYING // 11: 両者生存 (ここには来ない)
        };
        status_ = kNextStatus[alive_mask];
    }

    // 5. おじゃま処理
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto& next = p.action_histories[(frame_ + 1) & 255];
        if (next.action.type == ActionType::NONE && p.active_ojama > 0) {
            auto& current = p.action_histories[frame_ & 255];
            if (current.action.type != ActionType::OJAMA) {
                next = {Action{ActionType::OJAMA}, 0};
            }
        }
    }

    // 0. フレーム遷移
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        auto& next = p.action_histories[(frame_ + 1) & 255];
        if (next.action.type == ActionType::NONE) {
            auto& current = p.action_histories[frame_ & 255];
            if (current.action.type != ActionType::PASS) {
                ++(p.active_next_pos);
            }
        }
    }

    // Clear current frame history for next cycle (to ensure it represents "empty" 256 frames later)
    for (int id = 0; id < 2; ++id) {
        players_[id].action_histories[frame_ & 255] = {};
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

int PuyotanMatch::nextInt(int32_t& seed, int max) {
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 15);
    // JS does `this.y >>> 0`, which casts the bit pattern to uint32_t.
    uint32_t r = static_cast<uint32_t>(seed);
    return static_cast<int>(r % static_cast<uint32_t>(max));
}

} // namespace puyotan
