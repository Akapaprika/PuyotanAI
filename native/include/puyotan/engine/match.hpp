#pragma once

#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/common/types.hpp>
#include <string>
#include <optional>
#include <array>

namespace puyotan {

/**
 * @struct PuyotanPlayer
 * @brief Encapsulates a single player's game state, including their board and scoring.
 */
struct PuyotanPlayer {
    Board field;                        ///< 6x14 BitBoard-based playing field
    ActionState current_action{};       ///< Action being processed in the current frame
    ActionState next_action{};          ///< Action scheduled for the next frame

    int32_t active_next_pos = 0;        ///< Current index into the Tsumo (sequence of pieces)
    int score = 0;                      ///< Cumulative raw score
    int used_score = 0;                 ///< Score already converted into Ojama puyos
    uint16_t non_active_ojama = 0;      ///< Incoming Ojama not yet "active" (can be offset)
    uint16_t active_ojama = 0;          ///< Ojama puyos ready to fall on the board
    uint8_t chain_count = 0;            ///< current active chain length (0 if not chaining)
    uint8_t last_chain_count = 0;       ///< Result of the most recently finished chain
    uint16_t total_ojama_dropped = 0;   ///< Total ojama puyos that have ever fallen on this field

    // -----------------------------------------------------------------------
    // Per-turn event flags (set by engine, consumed by reward layer / GUI)
    // Reset at the start of each PUT action.
    // -----------------------------------------------------------------------
    bool     last_all_clear    = false; ///< True if the field was fully cleared this chain sequence
    uint16_t last_erased_count = 0;     ///< Total puyos erased in the last completed chain sequence
    
    PuyotanPlayer() = default;

    /**
     * @brief Drops a specific number of Ojama puyos onto the player's board.
     * @param num Number of Ojama puyos to drop.
     * @param seed Reference to the match RNG seed for column distribution.
     */
    void fallOjama(int num, uint32_t& seed) noexcept;
};

/**
 * @class PuyotanMatch
 * @brief Orchestrates a Puyo Puyo match between two players.
 * 
 * Handles frame-by-frame simulation, action processing, Tsumo (piece) management,
 * and Ojama (nuisance) distribution.
 */
class PuyotanMatch {
public:
    /**
     * @brief Constructs a new match with a specific RNG seed for Tsumo sequences.
     * @param seed The random seed (must be non-zero).
     */
    explicit PuyotanMatch(uint32_t seed = 1u) noexcept;
    PuyotanMatch(const PuyotanMatch&) = default;
    PuyotanMatch& operator=(const PuyotanMatch&) = default;

    /** @brief Transitions the match from Ready to Playing. */
    void start() noexcept;

    /**
     * @brief Assigns an action to a specific player for the current decision point.
     * @param player_id ID of the player (0 or 1).
     * @param action The action to perform.
     * @return True if the action was valid and accepted.
     */
    bool setAction(int player_id, Action action) noexcept;

    /**
     * @brief Checks if all required actions have been provided to advance the frame.
     * @return True if the simulation can proceed.
     */
    bool canStepNextFrame() const noexcept;

    /**
     * @brief Advances the match simulation by exactly one frame.
     * Processes gravity, chains, and action transitions.
     */
    void stepNextFrame() noexcept;

    /** @brief Returns a reference to the specified player's state. */
    const PuyotanPlayer& getPlayer(int id) const noexcept { return players_[id]; }

    /** @brief Returns a PuyoPiece from the Tsumo sequence at an offset from player's current position. */
    PuyoPiece getPiece(int player_id, int index_offset) const noexcept {
        return const_cast<Tsumo&>(tsumo_).get(players_[player_id].active_next_pos + index_offset);
    }

    /** @brief Returns the Tsumo generator used by the match. */
    const Tsumo& getTsumo() const noexcept { return tsumo_; }

    /** @brief Returns the current total frame count since the start of the match. */
    int32_t getFrame() const noexcept { return frame_; }

    /** @brief Returns the overall status (Playing, WinP1, etc.). */
    MatchStatus getStatus() const noexcept { return status_; }

    /**
     * Returns a bitmask of player IDs that need a human PUT decision.
     * Bit 0 = Player 0, Bit 1 = Player 1.
     *   0  -> No decisions needed; all players are processing auto-frames (chain, ojama, etc.)
     *   1  -> Player 0 needs to confirm their PUT
     *   2  -> Player 1 needs to confirm their PUT
     *   3  -> Both players need to confirm their PUT
     * The engine differentiates genuine decision points (ActionType::None) from 
     * automatic internal frames (CHAIN, CHAIN_FALL, OJAMA), which the engine drives itself.
     */
    int getDecisionMask() const noexcept;

    /**
     * @brief Performs a high-speed batch simulation of multiple games.
     * 
     * Uses a deterministic internal move pattern to stress-test the engine
     * and measure raw throughput (FPS).
     * @param num_games Number of matches to simulate.
     * @param seed Base RNG seed.
     * @return Total frames processed across all games.
     */
    static int64_t runBatch(int num_games, uint32_t seed) noexcept;

    /**
     * @brief Fast-forwards the simulation until a player input is required.
     * 
     * Skips over automatic frames (Chains, Falling, Ojama).
     * @return Bitmask of players needing ActionType::Put (1:P1, 2:P2, 3:Both).
     */
    int stepUntilDecision() noexcept;

    /**
     * @brief PCG-like simple Xorshift RNG for internal engine use (e.g. Ojama positions).
     * @param seed Reference to the 32-bit state.
     * @param max Exclusive upper bound.
     * @return Random integer in range [0, max-1].
     */
    static int nextInt(uint32_t& seed, int max) noexcept;

private:
    uint32_t seed_;
    Tsumo tsumo_;
    PuyotanPlayer players_[config::Rule::kNumPlayers];
    int32_t frame_ = 1;
    MatchStatus status_ = MatchStatus::Ready;

    void sendOjama(int sender_id, int ojama) noexcept;
    void activateOjama(int finishing_player_id) noexcept;

    // Pre-computed chain group data, cached between CHAIN_FALL/PUT → CHAIN turns.
    // Stored here (not in PuyotanPlayer) to keep PuyotanPlayer compact and cache-friendly.
    std::array<ErasureData, config::Rule::kNumPlayers> pending_erasure_;
};

} // namespace puyotan
