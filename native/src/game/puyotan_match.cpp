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

PuyotanMatch::PuyotanMatch(int32_t seed) : seed_(seed == 0 ? 1 : seed), tsumo_(seed == 0 ? 1 : seed) {
    for (auto& p : players_) {
        p = PuyotanPlayer();
    }
}

void PuyotanMatch::start() {
    if (frame_ == 0) {
        ++frame_;
        status_ = MatchStatus::PLAYING;
    }
}

std::string PuyotanMatch::getStatusText() const {
    switch (status_) {
        case MatchStatus::READY: return "READY";
        case MatchStatus::PLAYING: return "PLAYING";
        case MatchStatus::WIN_P1: return "1P WIN";
        case MatchStatus::WIN_P2: return "2P WIN";
        case MatchStatus::DRAW: return "DRAW";
        default: return "UNKNOWN";
    }
}

bool PuyotanMatch::setAction(int id, Action action) {
    if (frame_ <= 0) return false;
    auto& p = players_[id];
    if (p.current_action.action.type == ActionType::NONE) {
        switch (action.type) {
            case ActionType::PASS:
                p.current_action = {action, 0};
                return true;
            case ActionType::PUT:
                p.current_action = {action, 1};
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
        if (players_[id].current_action.action.type == ActionType::NONE) return false;
    }
    return true;
}

void PuyotanMatch::stepNextFrame() {
    if (!canStepNextFrame()) return;

    // 1. 行動選択・予約
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        if (p.current_action.action.type != ActionType::NONE && p.current_action.remaining_frame > 0) {
            p.next_action = {p.current_action.action, static_cast<uint8_t>(p.current_action.remaining_frame - 1)};
        }
    }

    // 2. 行動実行
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        if (p.current_action.action.type != ActionType::NONE && p.current_action.remaining_frame == 0) {
            const auto& action = p.current_action.action;
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
                        p.next_action = {Action{ActionType::CHAIN}, 1};
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
                        p.next_action = {Action{ActionType::CHAIN_FALL}, 0};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::CHAIN_FALL: {
                    uint8_t dirty_colors = Gravity::execute(p.field);
                    if (Chain::canFire(p.field, dirty_colors)) {
                        p.next_action = {Action{ActionType::CHAIN}, 1};
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

    // 3. 窒息判定 (Branchless Status Map)
    uint32_t alive_mask = 0;
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        bool is_alive = p.next_action.action.type != ActionType::NONE || 
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

    // 4. おじゃま処理
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        if (p.next_action.action.type == ActionType::NONE && p.active_ojama > 0) {
            if (p.current_action.action.type != ActionType::OJAMA) {
                p.next_action = {Action{ActionType::OJAMA}, 0};
            }
        }
    }

    // 5. ツモ・フレーム遷移
    for (int id = 0; id < 2; ++id) {
        auto& p = players_[id];
        if (p.next_action.action.type == ActionType::NONE) {
            if (p.current_action.action.type != ActionType::PASS) {
                ++(p.active_next_pos);
            }
        }
    }

    // Advance actions: current = next, and clear next for the next step.
    for (int id = 0; id < 2; ++id) {
        players_[id].current_action = players_[id].next_action;
        players_[id].next_action = {};
    }

    ++frame_;
}

void PuyotanMatch::sendOjama(int sender_id, int ojama) {
    int target_id = 1 - sender_id;
    players_[target_id].non_active_ojama += ojama;
}

void PuyotanMatch::activateOjama(int finishing_player_id) {
    int target_id = 1 - finishing_player_id;
    auto& p = players_[target_id];
    p.active_ojama += p.non_active_ojama;
    p.non_active_ojama = 0;
}

int PuyotanMatch::stepUntilDecision() {
    while (status_ == MatchStatus::PLAYING) {
        int mask = 0;
        if (players_[0].current_action.action.type == ActionType::NONE) mask |= 1;
        if (players_[1].current_action.action.type == ActionType::NONE) mask |= 2;

        if (mask != 0) return mask;

        if (canStepNextFrame()) {
            stepNextFrame();
        } else {
            // This case should be covered by mask check, but failsafe.
            return 0;
        }
    }
    return 0;
}

int64_t PuyotanMatch::runBatch(int num_games, int32_t seed) {
    int64_t total_frames = 0;
    
    // 6 at col 5, 6 at 4, 6 at 3, etc.
    const int move_plan[] = {
        5,5,5,5,5,5,
        4,4,4,4,4,4,
        3,3,3,3,3,3
    };
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);

    for (int i = 0; i < num_games; ++i) {
        PuyotanMatch match(seed + i);
        match.start();

        int p1_move = 0;
        int p2_move = 0;

        while (match.getStatus() == MatchStatus::PLAYING) {
            bool action_set = false;
            if (match.players_[0].current_action.action.type == ActionType::NONE) {
                int col = (p1_move < num_moves) ? move_plan[p1_move] : 2;
                if (match.setAction(0, Action{ActionType::PUT, static_cast<int8_t>(col), Rotation::Up})) {
                    p1_move++;
                    action_set = true;
                }
            }

            if (match.players_[1].current_action.action.type == ActionType::NONE) {
                int col = (p2_move < num_moves) ? move_plan[p2_move] : 2;
                if (match.setAction(1, Action{ActionType::PUT, static_cast<int8_t>(col), Rotation::Up})) {
                    p2_move++;
                    action_set = true;
                }
            }

            if (match.canStepNextFrame()) {
                match.stepNextFrame();
                total_frames++;
            } else if (!action_set) {
                // If we can't step and we didn't just set an action, we are stuck.
                // This shouldn't happen with the current engine logic, but let's be safe.
                break; 
            }
            
            // Failsafe
            if (match.frame_ > 3000) break;
        }
    }
    return total_frames;
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
