#pragma once
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <string>

namespace puyotan {

struct alignas(64) PuyotanPlayer {
    Board field;
    ActionState current_action{};
    int32_t active_next_pos = 0;
    int score = 0;
    int used_score = 0;
    uint16_t non_active_ojama = 0;
    uint16_t active_ojama = 0;
    uint8_t chain_count = 0;
    PuyotanPlayer() = default;

    void fallOjama(int num, uint32_t& seed) noexcept;
};

class PuyotanMatch {
  public:
    // 通常対局用（内部で TsumoSequence を生成・所有）
    explicit PuyotanMatch(uint32_t seed = 1u) noexcept;

    // AI探索用：すでに生成された TsumoSequence の参照を共有する（超高速コンストラクタ）
    explicit PuyotanMatch(const TsumoSequence* sequence) noexcept;

    PuyotanMatch(const PuyotanMatch&) = default;
    PuyotanMatch& operator=(const PuyotanMatch&) = default;

    void start() noexcept;

    __forceinline bool setAction(int player_id, Action action) noexcept {
        assert(status_ == MatchStatus::Playing &&
               "Cannot set action to match not in PLAYING status");
        auto& p = players_[player_id];
        assert(p.current_action.action.type == ActionType::None &&
               "Action already set for this player in this turn");
        switch (action.type) {
            case ActionType::Pass:
                p.current_action = {action, 0};
                return true;
            case ActionType::Put:
                p.current_action = {action, 1};
                return true;
            default:
                return false;
        }
    }

    __forceinline bool canStepNextFrame() const noexcept {
        const int playing = static_cast<int>(status_ == MatchStatus::Playing);
        const int p0_ready = static_cast<int>(players_[0].current_action.action.type != ActionType::None);
        const int p1_ready = static_cast<int>(players_[1].current_action.action.type != ActionType::None);
        return (playing & p0_ready & p1_ready) != 0;
    }

    void stepNextFrame() noexcept;
    __forceinline void stepPlayerFrame(int id, const std::array<ActionType, 2>& prev_types) noexcept;

    const PuyotanPlayer& getPlayer(int id) const noexcept {
        return players_[id];
    }

    PuyoPiece getPiece(int player_id, int index_offset) const noexcept {
        assert(tsumo_sequence_ != nullptr);
        return tsumo_sequence_->get(players_[player_id].active_next_pos + index_offset);
    }

    const TsumoSequence* getTsumoSequence() const noexcept {
        return tsumo_sequence_;
    }

    int32_t getFrame() const noexcept {
        return frame_;
    }

    MatchStatus getStatus() const noexcept {
        return status_;
    }

    int getDecisionMask() const noexcept;
    static int64_t runBatch(int num_games, uint32_t seed) noexcept;
    int stepUntilDecision() noexcept;
    static int nextInt(uint32_t& seed, int max) noexcept;

  private:
    uint32_t seed_ = 0u;
    std::shared_ptr<const TsumoSequence> owned_sequence_; ///< 自身で所有する場合のみ使用
    const TsumoSequence* tsumo_sequence_ = nullptr;       ///< 実際のツモアクセス用ポインタ (8B)
    PuyotanPlayer players_[config::Rule::kNumPlayers];
    int32_t frame_ = 1;
    MatchStatus status_ = MatchStatus::Ready;

    void sendOjama(int sender_id, int ojama) noexcept;
    void activateOjama(int finishing_player_id) noexcept;

    std::array<ErasureData, config::Rule::kNumPlayers> pending_erasure_;
};

} // namespace puyotan