#pragma once

#include <puyotan/core/board.hpp>
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
    Board field;                                     // 96 bytes (aligned 16)
    std::array<ActionState, 256> action_histories{}; // frame & 255 -> state (NONE type if empty)

    // Grouping for 16-byte alignment
    int score = 0;              // 4 bytes
    int used_score = 0;         // 4 bytes
    uint16_t active_next_pos = 0; // 2 bytes (0-1000)
    uint16_t non_active_ojama = 0; // 2 bytes
    uint16_t active_ojama = 0;     // 2 bytes
    uint8_t chain_count = 0;       // 1 byte
    uint8_t padding = 0;           // 1 byte (explict for 16-byte block)

    void fallOjama(int num, int32_t& seed);
};

/**
 * Puyotan frame-based match manager.
 */
class PuyotanMatch {
public:
    explicit PuyotanMatch(int32_t seed = 0);

    void start();
    bool setAction(int player_id, Action action);
    bool canStepNextFrame() const;
    void stepNextFrame();

    const PuyotanPlayer& getPlayer(int id) const { return players_[id]; }
    PuyoPiece getPiece(int player_id, int index_offset) const {
        return tsumo_.get(players_[player_id].active_next_pos + index_offset);
    }
    int getFrame() const { return frame_; }
    MatchStatus getStatus() const { return status_; }
    std::string getStatusText() const;

    // Helper to get random int for ojama fall positions
    static int nextInt(int32_t& seed, int max);

private:
    int32_t seed_;
    Tsumo tsumo_;
    PuyotanPlayer players_[2];
    int frame_ = 0;
    MatchStatus status_ = MatchStatus::READY;

    // calculateScore: removed (was declared but never implemented)
    void sendOjama(int sender_id, int ojama);
    void activateOjama(int sender_id);
};

} // namespace puyotan
