#pragma once

#include <puyotan/core/board.hpp>
#include <puyotan/game/tsumo.hpp>

namespace puyotan {

/**
 * Simulator
 *   High-level game simulation integrating board, tsumo, and rules.
 */
class Simulator {
public:
    explicit Simulator(uint32_t seed = 1u);

    /**
     * Executes one move: places a piece, applies gravity, and processes chains.
     * @return The score gained from this single move (for RL reward shaping).
     */
    int step(int x, Rotation rotation);

    /**
     * Checks if the death condition is met.
     */
    bool isGameOver() const noexcept { return is_game_over_; }

    /**
     * Resets the game state.
     */
    void reset(uint32_t seed);

    const Board& getBoard() const noexcept { return board_; }
    const Tsumo& getTsumo() const noexcept { return tsumo_; }
    int getTsumoIndex() const noexcept { return tsumo_index_; }
    int getTotalScore() const noexcept { return total_score_; }
    
    PuyoPiece getCurrentPiece();

    /**
     * Runs num_games full games in pure C++ using the benchmark move pattern:
     *   - 6 moves at col 5, 6 at col 4, 6 at col 3, then col 2 until game over.
     * Returns total steps executed (for throughput calculation).
     */
    int64_t runBatch(int num_games, uint32_t seed);

private:
    Board board_;
    Tsumo tsumo_;
    int tsumo_index_ = 0;
    int total_score_ = 0;
    bool is_game_over_ = false;

    void updateGameOver();
};

} // namespace puyotan
