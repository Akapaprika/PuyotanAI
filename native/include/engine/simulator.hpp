#pragma once

#include "engine/board.hpp"
#include "engine/tsumo.hpp"

namespace puyotan {

/**
 * Simulator
 *   Integrates Board, Tsumo, and game rules to simulate a full game.
 */
class Simulator {
public:
    /**
     * @param seed Random seed for Tsumo generation.
     */
    explicit Simulator(uint32_t seed = 0);

    /**
     * Executes one move: places a piece, applies gravity, and processes chains.
     * @param x The column to place the axis puyo (0..5).
     * @param direction The rotation of the sub puyo (0: up, 1: right, 2: down, 3: left).
     */
    void step(int x, int direction);

    /**
     * Checks if the death condition is met (spawn point obscured).
     */
    bool isGameOver() const { return is_game_over_; }

    /**
     * Resets the game state.
     */
    void reset(uint32_t seed);

    // ---- Accessors ----
    const Board& getBoard() const { return board_; }
    const Tsumo& getTsumo() const { return tsumo_; }
    int getTsumoIndex() const { return tsumo_index_; }
    int getTotalScore() const { return total_score_; }
    
    /**
     * Returns the piece at the current tsumo index.
     */
    PuyoPiece getCurrentPiece() const;

private:
    Board board_;
    Tsumo tsumo_;
    int tsumo_index_ = 0;
    int total_score_ = 0;
    bool is_game_over_ = false;

    /**
     * Updates is_game_over_ based on the death condition.
     */
    void updateGameOver();
};

} // namespace puyotan
