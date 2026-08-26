#pragma once
#include <array>
#include <cassert>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/engine/tsumo.hpp>

namespace puyotan {

/**
 * @struct PuyotanPlayer
 * @brief Encapsulates a single player's game state, including their board and scoring.
 */
struct alignas(64) PuyotanPlayer {
    Board field;                       ///< 6x14 BitBoard-based playing field
    ActionState current_action{};      ///< Action being processed in the current frame
    int32_t active_next_pos = 0;       ///< Current index into the Tsumo
    int score = 0;                     ///< Cumulative raw score
    int used_score = 0;                ///< Score already converted into Ojama puyos
    uint16_t non_active_ojama = 0;     ///< Incoming Ojama not yet "active"
    uint16_t active_ojama = 0;         ///< Ojama puyos ready to fall on the board
    uint8_t chain_count = 0;           ///< Current active chain length
    PuyotanPlayer() = default;

    void fallOjama(int num, uint32_t& seed) noexcept;
};

/**
 * @class PuyotanMatch
 * @brief Orchestrates a Puyo Puyo match between two players.
 */
class PuyotanMatch {
  public:
    explicit PuyotanMatch(uint32_t seed = 1u) noexcept;
    PuyotanMatch(const PuyotanMatch&) = default;
    PuyotanMatch& operator=(const PuyotanMatch&) = default;

    void start() noexcept;
    bool setAction(int player_id, Action action) noexcept;
    [[nodiscard]] bool canStepNextFrame() const noexcept;
    void stepNextFrame() noexcept;
    int stepUntilDecision() noexcept;

    [[nodiscard]] const PuyotanPlayer& getPlayer(int id) const noexcept {
        return players_[id];
    }
    [[nodiscard]] PuyoPiece getPiece(int player_id, int index_offset) const noexcept;
    [[nodiscard]] const Tsumo& getTsumo() const noexcept {
        return tsumo_;
    }
    [[nodiscard]] int32_t getFrame() const noexcept {
        return frame_;
    }
    [[nodiscard]] MatchStatus getStatus() const noexcept {
        return status_;
    }
    [[nodiscard]] int getDecisionMask() const noexcept;

    static int nextInt(uint32_t& seed, int max) noexcept;

  private:
    uint32_t seed_ = 0u;
    Tsumo tsumo_;
    PuyotanPlayer players_[config::Rule::kNumPlayers];
    int32_t frame_ = 1;
    MatchStatus status_ = MatchStatus::Ready;

    void stepPlayerFrame(int id) noexcept;
    void sendOjama(int sender_id, int ojama) noexcept;
    void activateOjama(int finishing_player_id) noexcept;

    std::array<ErasureData, config::Rule::kNumPlayers> pending_erasure_;
};
} // namespace puyotan
