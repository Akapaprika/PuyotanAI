#pragma once

#include <puyotan/core/board.hpp>
#include <puyotan/game/tsumo.hpp>
#include <puyotan/common/types.hpp>
#include <vector>
#include <string>
#include <optional>
#include <map>

namespace puyotan {

/**
 * Player state for Puyotan frame-based match.
 */
struct PuyotanPlayer {
    Board field;
    std::map<int, ActionState> action_histories; // frame -> state
    int active_next_pos = 0;
    int score = 0;
    int used_score = 0;
    int non_active_ojama = 0;
    int active_ojama = 0;
    int chain_count = 0;

    void fallOjama(int num, uint32_t& seed);
};

/**
 * Puyotan frame-based match manager.
 */
class PuyotanMatch {
public:
    explicit PuyotanMatch(uint32_t seed = 0);

    void start();
    bool setAction(int player_id, Action action);
    bool canStepNextFrame() const;
    void stepNextFrame();

    const PuyotanPlayer& getPlayer(int id) const { return players_[id]; }
    int getFrame() const { return frame_; }
    MatchStatus getStatus() const { return status_; }
    std::string getStatusText() const;

private:
    uint32_t seed_;
    Tsumo tsumo_;
    PuyotanPlayer players_[2];
    int frame_ = 0;
    MatchStatus status_ = MatchStatus::READY;

    int calculateScore(int num, int colors, const std::vector<int>& group_sizes, int chain);
    void sendOjama(int sender_id, int ojama);
    void activateOjama(int sender_id);
    
    // Helper to get random int for ojama fall
    int nextInt(uint32_t& seed, int max);
};

} // namespace puyotan
