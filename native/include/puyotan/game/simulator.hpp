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
    explicit Simulator(uint32_t seed = 0);

    /**
     * Executes one move: places a piece, applies gravity, and processes chains.
     */
    void step(int x, Rotation rotation);

    /**
     * Checks if the death condition is met.
     */
    bool isGameOver() const { return is_game_over_; }

    /**
     * Resets the game state.
     */
    void reset(uint32_t seed);

    const Board& getBoard() const { return board_; }
    const Tsumo& getTsumo() const { return tsumo_; }
    int getTsumoIndex() const { return tsumo_index_; }
    int getTotalScore() const { return total_score_; }
    
    PuyoPiece getCurrentPiece() const;

private:
    Board board_;
    Tsumo tsumo_;
    int tsumo_index_ = 0;
    int total_score_ = 0;
    bool is_game_over_ = false;

    void updateGameOver();
};

} // namespace puyotan
