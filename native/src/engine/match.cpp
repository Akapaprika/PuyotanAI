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
        int drop_num;

        if (num >= width) {
            // 1段丸ごと降る場合
            // 単純に各列の高さに直接1個ずつおじゃまを設置して、高さを積み上げるだけ（超高速）。
            for (int x = 0; x < width; ++x) {
                const int h = field.getColumnHeight(x);
                field.dropNewPiece(x, h, Cell::Ojama);
            }
            drop_num = width;
        } else {
            // 5個以下の端数が降る場合
            uint32_t mask = 0;
            for (int i = 0; i < num; ++i) {
                const int pos = PuyotanMatch::nextInt(seed, width - i);
                const uint32_t free = ~mask & 0x3F;
                const uint32_t chosen_bit = kOjamaColumnLut[free][pos];
                mask |=
                    chosen_bit; // 次のループで重複を避けるためにビットを記録

                // 決定された列（1ビット）から、TZCNT命令（std::countr_zero）で列番号を取得し、
                // その列の高さに直接おじゃまを設置する
                const int x = std::countr_zero(chosen_bit);
                const int h = field.getColumnHeight(x);
                field.dropNewPiece(x, h, Cell::Ojama);
            }
            drop_num = num;
        }

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
}

void PuyotanMatch::start() noexcept {
    assert(status_ == MatchStatus::Ready &&
           "start() should only be called once when match is ready");
    status_ = MatchStatus::Playing;
}

void PuyotanMatch::stepNextFrame() noexcept {
    if (!canStepNextFrame())
        return;

    std::array<ActionType, config::Rule::kNumPlayers> prev_types;
    prev_types[0] = players_[0].current_action.action.type;
    prev_types[1] = players_[1].current_action.action.type;

    stepPlayerFrame(0, prev_types);
    stepPlayerFrame(1, prev_types);

    // 3. Death check (ループを廃止してプレイヤー 0 と 1 を個別にベタ書き)
    uint32_t alive_mask = 0;
    {
        // プレイヤー0
        bool is_alive0 =
            (players_[0].current_action.action.type != ActionType::None) |
            !players_[0].field.isOccupied(config::Rule::kDeathCol,
                                          config::Rule::kDeathRow);
        alive_mask |= (static_cast<uint32_t>(is_alive0) << 0);

        // プレイヤー1
        bool is_alive1 =
            (players_[1].current_action.action.type != ActionType::None) |
            !players_[1].field.isOccupied(config::Rule::kDeathCol,
                                          config::Rule::kDeathRow);
        alive_mask |= (static_cast<uint32_t>(is_alive1) << 1);
    }

    if (alive_mask != 3) [[unlikely]] {
        static constexpr MatchStatus kNextStatus[] = {
            MatchStatus::Draw, MatchStatus::WinP1, MatchStatus::WinP2,
            MatchStatus::Playing};
        status_ = kNextStatus[alive_mask];
    }

    // 4 & 5. Post-turn processing (こちらもループを廃止して個別処理)
    // プレイヤー0
    if (players_[0].current_action.action.type == ActionType::None) {
        if (players_[0].active_ojama > 0 &&
            prev_types[0] != ActionType::Ojama) {
            players_[0].current_action = {Action{ActionType::Ojama}, 0};
        } else if (prev_types[0] != ActionType::Pass) {
            ++players_[0].active_next_pos;
        }
    }
    // プレイヤー1
    if (players_[1].current_action.action.type == ActionType::None) {
        if (players_[1].active_ojama > 0 &&
            prev_types[1] != ActionType::Ojama) {
            players_[1].current_action = {Action{ActionType::Ojama}, 0};
        } else if (prev_types[1] != ActionType::Pass) {
            ++players_[1].active_next_pos;
        }
    }

    ++frame_;
}

// プレイヤー1人分の個別ステップ関数を実装
__forceinline void PuyotanMatch::stepPlayerFrame(
    int id, const std::array<ActionType, 2>& prev_types) noexcept {
    auto& p = players_[id];

    if (p.current_action.remaining_frame > 0) {
        p.current_action.remaining_frame--;
    } else {
        const auto& action = p.current_action.action;
        switch (action.type) {
            case ActionType::Pass:
                p.current_action = {};
                break;
            case ActionType::Put: {
                const PuyoPiece tumo = tsumo_.get(p.active_next_pos);
                
                // 従来の getColumnHeight や dropNewPiece の複数回呼び出しを、上記関数1回に集約！
                int h_axis = 0;
                int h_sub = 0;
                p.field.dropPiecePair(action.x, action.rotation, tumo.axis, tumo.sub, h_axis, h_sub);
            
                // スコア計算（h_axis, h_sub がすでに取得できているためそのまま利用）
                p.score += std::max(0, config::Board::kSpawnRow - std::max(h_axis, h_sub));
            
                const uint32_t dirty_colors = tumo.dirty_flag;
                Chain::scanGroups(p.field, pending_erasure_[id], dirty_colors);
            
                if (pending_erasure_[id].num_erased > 0) {
                    p.current_action = {Action{ActionType::Chain}, 1};
                } else {
                    p.current_action = {};
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
                    p.current_action = {};
                }
                break;
            }
            case ActionType::ChainFall: {
                uint32_t dirty_colors = Gravity::execute(p.field);
                Chain::scanGroups(p.field, pending_erasure_[id], dirty_colors);
                if (pending_erasure_[id].num_erased > 0) {
                    p.current_action = {Action{ActionType::Chain}, 1};
                } else {
                    p.chain_count = 0;
                    activateOjama(id);
                    p.current_action = {};
                }
                break;
            }
            case ActionType::Ojama: {
                int fall_num = std::min(static_cast<int>(p.active_ojama),
                                        config::Rule::kMaxOjamaPerFall);
                p.active_ojama -= static_cast<uint16_t>(fall_num);
                // 実際におじゃまが降る最初のフレームで、初めて getSeed()
                // を呼ぶ（遅延評価）
                if (seed_ == 0u) [[unlikely]] {
                    seed_ = tsumo_.getSeed();
                }

                p.fallOjama(fall_num, seed_);
                p.current_action = {};
                break;
            }
            default:
                break;
        }
    }
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

inline int fast_modulo(uint32_t val, int max) noexcept {
    struct Magic {
        uint64_t mul;
        uint8_t shift;
    };

    // 32ビット符号なし整数 [0, 2^32-1] の全域で、
    // (val * mul) >> shift が val / d と厳密に一致する定数テーブル
    static constexpr Magic kMagic[7] = {
        {0, 0},
        {1ULL, 0},             // d = 1 -> (val * 1) >> 0
        {1ULL, 1},             // d = 2 -> (val * 1) >> 1
        {0xAAAAAAABULL, 33},  // d = 3 -> (val * 0xAAAAAAAB) >> 33
        {1ULL, 2},             // d = 4 -> (val * 1) >> 2
        {0xCCCCCCCDULL, 34},  // d = 5 -> (val * 0xCCCCCCCD) >> 34
        {0xAAAAAAABULL, 34}   // d = 6 -> (val * 0xAAAAAAAB) >> 34 (33 + 1)
    };

    assert(max >= 1 && max <= 6);

    // 1回の乗算・1回のシフト・1回の積和演算だけで完結（完全分岐レス、定数時間で実行可能）
    const uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(val) * kMagic[max].mul) >> kMagic[max].shift);
    return static_cast<int>(val - quotient * max);
}

int PuyotanMatch::nextInt(uint32_t& seed, int max) noexcept {
    assert(seed != 0u);
    seed ^= (seed << 13);
    seed ^= static_cast<uint32_t>(static_cast<int32_t>(seed) >> 17);
    seed ^= (seed << 15);
    return fast_modulo(seed, max);
}
} // namespace puyotan
