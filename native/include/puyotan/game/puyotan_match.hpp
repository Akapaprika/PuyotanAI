#pragma once

#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/types.hpp>
#include <string>
#include <optional>
#include <array>

namespace puyotan {

/**
 * Player state for Puyotan frame-based match.
 */
struct PuyotanPlayer {
    Board field;                // 96 bytes (aligned 16)
    ActionState current_action{};
    ActionState next_action{};

    // Grouping for 16-byte alignment
    int32_t active_next_pos = 0; // 4 bytes
    int score = 0;              // 4 bytes
    int used_score = 0;         // 4 bytes
    uint16_t non_active_ojama = 0; // 2 bytes
    uint16_t active_ojama = 0;     // 2 bytes
    uint8_t chain_count = 0;       // 1 byte
    uint8_t padding = 0;           // 1 byte (explict for 16-byte block)

    PuyotanPlayer() = default;
    void fallOjama(int num, uint32_t& seed);
};

/**
 * Puyotan frame-based match manager.
 */
class PuyotanMatch {
public:
    explicit PuyotanMatch(uint32_t seed = 1u);
    PuyotanMatch(const PuyotanMatch&) = default;
    PuyotanMatch& operator=(const PuyotanMatch&) = default;

    void start();
    bool setAction(int player_id, Action action);
    bool canStepNextFrame() const;
    void stepNextFrame();

    const PuyotanPlayer& getPlayer(int id) const noexcept { return players_[id]; }
    PuyoPiece getPiece(int player_id, int index_offset) const noexcept {
        return const_cast<Tsumo&>(tsumo_).get(players_[player_id].active_next_pos + index_offset);
    }
    const Tsumo& getTsumo() const noexcept { return tsumo_; }
    int32_t getFrame() const noexcept { return frame_; }
    MatchStatus getStatus() const noexcept { return status_; }

     /**
     * Runs num_games full matches in pure C++ using the benchmark move pattern
     * for both players. Returns total frames executed.
     */
    static int64_t runBatch(int num_games, uint32_t seed);

    /**
     * Steps the match until at least one player needs to make a decision
     * (ActionType::NONE in current frame history).
     * Returns a bitmask of player IDs that need actions (1: P1, 2: P2, 3: Both),
     * or 0 if game over or error.
     */
    int stepUntilDecision();

    // Helper to get random int for ojama fall positions
    static int nextInt(uint32_t& seed, int max) noexcept;

private:
    uint32_t seed_;
    Tsumo tsumo_;
    PuyotanPlayer players_[config::Rule::kNumPlayers];
    int32_t frame_ = 1;
    MatchStatus status_ = MatchStatus::READY;

    void sendOjama(int sender_id, int ojama);
    void activateOjama(int finishing_player_id);

    // Pre-computed chain group data, cached between CHAIN_FALL/PUT → CHAIN turns.
    // Stored here (not in PuyotanPlayer) to keep PuyotanPlayer compact and cache-friendly.
    std::array<ErasureData, config::Rule::kNumPlayers> pending_erasure_;
};

} // namespace puyotan
