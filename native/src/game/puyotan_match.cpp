#include <puyotan/game/puyotan_match.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>

namespace puyotan {

void PuyotanPlayer::fallOjama(int num, uint32_t& seed) noexcept {
    constexpr int width = config::Board::kWidth;
    while (num > 0) {
        if (num >= width) {
            field.setRowMask(config::Board::kSpawnRow, Cell::Ojama, 0x3F);
            Gravity::execute(field);
            num -= width;
        } else {
            uint32_t mask = 0;
            for (int i = 0; i < num; ++i) {
                const int pos = PuyotanMatch::nextInt(seed, width - i);
                uint32_t free = ~mask & 0x3F;
                for (int j = 0; j < pos; ++j) {
                    free &= (free - 1); // BLSR: Clear the lowest set bit
                }
                mask |= (free & -free); // Extract the new lowest set bit
            }
            field.setRowMask(config::Board::kSpawnRow, Cell::Ojama, mask);
            Gravity::execute(field);
            break;
        }
    }
}

PuyotanMatch::PuyotanMatch(uint32_t seed) noexcept : seed_(seed), tsumo_(seed) {
    assert(seed != 0u);
}

void PuyotanMatch::start() noexcept {
    assert(status_ == MatchStatus::READY && "start() should only be called once when match is ready");
    status_ = MatchStatus::PLAYING;
}

bool PuyotanMatch::setAction(int id, Action action) noexcept {
    assert(status_ == MatchStatus::PLAYING && "Cannot set action to match not in PLAYING status");
    auto& p = players_[id];
    assert(p.current_action.action.type == ActionType::NONE && "Action already set for this player in this turn");

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

bool PuyotanMatch::canStepNextFrame() const noexcept {
    if (status_ != MatchStatus::PLAYING) return false;
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        if (players_[id].current_action.action.type == ActionType::NONE) return false;
    }
    return true;
}

void PuyotanMatch::stepNextFrame() noexcept {
    if (!canStepNextFrame()) return;

    // 1. Execute or reserve actions
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];
        if (p.current_action.remaining_frame > 0) {
            p.next_action = {p.current_action.action, static_cast<uint8_t>(p.current_action.remaining_frame - 1)};
        } else {
            const auto& action = p.current_action.action;
            switch (action.type) {
                case ActionType::PASS:
                    break;
                case ActionType::PUT: {
                    const PuyoPiece tumo = tsumo_.get(p.active_next_pos);
                    const int r = static_cast<int>(action.rotation);
                    const int x_axis = action.x;
                    const int x_sub  = x_axis + kSubDx[r];

                    // O(1) base height calculation
                    const int h_axis = p.field.getColumnHeight(x_axis);
                    const int h_sub  = p.field.getColumnHeight(x_sub);

                    // kSoftDropBonusPerGrid == 1: multiply eliminated (static_assert below)
                    p.score += (config::Board::kSpawnRow - std::max(h_axis, h_sub));
                    static_assert(config::Score::kSoftDropBonusPerGrid == 1, "Assumed 1 for multiply elision");

                    // Direct BitBoard bit set (1 clock each, bypasses Gravity)
                    p.field.dropNewPiece(x_axis, h_axis + kAxisDy[r], tumo.axis);
                    p.field.dropNewPiece(x_sub,  h_sub  + kSubDy_Simple[r], tumo.sub);

                    // Zero-overhead erasure check restricted to only the 2 deposited colors
                    const uint32_t dirty_colors = tumo.dirty_flag;
                    pending_erasure_[id] = Chain::findGroups(p.field, dirty_colors);
                    
                    if (pending_erasure_[id].num_erased > 0) {
                        p.chain_count = 0;
                        p.next_action = {Action{ActionType::CHAIN}, 1}; 
                    }
                    break;
                }
                case ActionType::CHAIN: {
                    Chain::applyErasure(p.field, pending_erasure_[id]);
                    const ErasureData& info = pending_erasure_[id];
                    ++p.chain_count;
                    p.score += Scorer::calculateStepScore(info, p.chain_count);
                    
                    int ojama = (p.score - p.used_score) / config::Score::kTargetScore;
                    p.used_score += ojama * config::Score::kTargetScore;
                    
                    // Branchless ojama offset: min clamps to 0 automatically when ojama == 0
                    int used_non = std::min(ojama, static_cast<int>(p.non_active_ojama));
                    p.non_active_ojama -= static_cast<uint16_t>(used_non);
                    ojama -= used_non;

                    int used_active = std::min(ojama, static_cast<int>(p.active_ojama));
                    p.active_ojama -= static_cast<uint16_t>(used_active);
                    ojama -= used_active;

                    if (ojama > 0) {
                        sendOjama(id, ojama);
                    }
                    
                    // All Clear check (Pure mathematical branchless)
                    p.score += static_cast<int>(p.field.getOccupied().empty()) * config::Score::kAllClearBonus;

                    if (Gravity::canFall(p.field)) {
                        p.next_action = {Action{ActionType::CHAIN_FALL}, 0};
                    } else {
                        activateOjama(id);
                    }
                    break;
                }
                case ActionType::CHAIN_FALL: {
                    uint32_t dirty_colors = Gravity::execute(p.field);
                    pending_erasure_[id] = Chain::findGroups(p.field, dirty_colors);
                    if (pending_erasure_[id].num_erased > 0) {
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

    // 3. Death check (Branchless Status Map)
    uint32_t alive_mask = 0;
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];
        bool is_alive = (p.next_action.action.type != ActionType::NONE) |
                        (p.field.get(config::Rule::kDeathCol, config::Rule::kDeathRow) == Cell::Empty);
        alive_mask |= (is_alive << id);
    }

    static_assert(config::Rule::kNumPlayers == 2, "Match status mapping explicitly assumes 2 players");
    if (alive_mask != 3) { // At least one player is dead
        static constexpr MatchStatus kNextStatus[] = {
            MatchStatus::DRAW,   // 00: 両者死亡
            MatchStatus::WIN_P1, // 01: P1生存・P2死亡
            MatchStatus::WIN_P2, // 10: P1死亡・P2生存
            MatchStatus::PLAYING // 11: 両者生存 (ここには来ない)
        };
        status_ = kNextStatus[alive_mask];
    }

    // 4 & 5. Post-turn processing (Ojama, Tsumo, and Action Advance)
    for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
        auto& p = players_[id];
        
        if (p.next_action.action.type == ActionType::NONE) {
            // 4. Ojama (garbage) processing: Fall garbage if available and didn't just fall
            if (p.active_ojama > 0 && p.current_action.action.type != ActionType::OJAMA) {
                p.next_action = {Action{ActionType::OJAMA}, 0};
            } 
            // 5. Tsumo and frame transition: Otherwise, pull the next piece (unless passing)
            else if (p.current_action.action.type != ActionType::PASS) {
                ++(p.active_next_pos);
            }
        }

        // Advance actions: current = next, and clear next for the next step.
        p.current_action = p.next_action;
        p.next_action = {};
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
    while (status_ == MatchStatus::PLAYING) {
        int mask = 0;
        for (int id = 0; id < config::Rule::kNumPlayers; ++id) {
            mask |= (static_cast<int>(players_[id].current_action.action.type == ActionType::NONE) << id);
        }

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




int64_t PuyotanMatch::runBatch(int num_games, uint32_t seed) noexcept {
    int64_t total_frames = 0;
    
    // 6 at col 5, 6 at 4, 6 at 3, etc.
    const int move_plan[] = {
        5,5,5,5,5,5,
        4,4,4,4,4,4,
        3,3,3,3,3,3
    };
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);

    for (int i = 0; i < num_games; ++i) {
        PuyotanMatch match(seed + static_cast<uint32_t>(i));
        match.start();

        int p1_move = 0;
        int p2_move = 0;

        while (match.getStatus() == MatchStatus::PLAYING) {
            bool action_set = false;
            if (match.players_[0].current_action.action.type == ActionType::NONE) {
                // Safe branchless arithmetic: always accesses valid index
                int safe_idx_1 = p1_move * (p1_move < num_moves);
                int col = (p1_move < num_moves) * move_plan[safe_idx_1] + (p1_move >= num_moves) * 2;
                if (match.setAction(0, Action{ActionType::PUT, static_cast<int8_t>(col), Rotation::Up})) {
                    ++p1_move;
                    action_set = true;
                }
            }

            if (match.players_[1].current_action.action.type == ActionType::NONE) {
                // Safe branchless arithmetic: always accesses valid index
                int safe_idx_2 = p2_move * (p2_move < num_moves);
                int col = (p2_move < num_moves) * move_plan[safe_idx_2] + (p2_move >= num_moves) * 2;
                if (match.setAction(1, Action{ActionType::PUT, static_cast<int8_t>(col), Rotation::Up})) {
                    ++p2_move;
                    action_set = true;
                }
            }

            if (match.canStepNextFrame()) {
                match.stepNextFrame();
                ++total_frames;
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

int PuyotanMatch::nextInt(uint32_t& seed, int max) noexcept {
    assert(seed != 0u);
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 15);
    return static_cast<int>(seed % static_cast<uint32_t>(max));
}

} // namespace puyotan
