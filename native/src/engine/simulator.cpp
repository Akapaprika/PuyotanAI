#include "engine/simulator.hpp"
#include "engine/gravity.hpp"
#include "engine/chain.hpp"
#include "config/engine_config.hpp"

namespace puyotan {

Simulator::Simulator(uint32_t seed) : tsumo_(seed) {
}

void Simulator::reset(uint32_t seed) {
    board_ = Board();
    tsumo_.setSeed(seed);
    tsumo_index_ = 0;
    is_game_over_ = false;
}

PuyoPiece Simulator::getCurrentPiece() const {
    return tsumo_.get(tsumo_index_);
}

void Simulator::step(int x, int direction) {
    if (is_game_over_) {
        return;
    }

    PuyoPiece piece = getCurrentPiece();

    // 1. Determine spawn positions
    // Axis is always at x.
    // Sub direction: 0:up, 1:right, 2:down, 3:left
    static constexpr int kDx[] = {0, 1, 0, -1};
    int x_axis = x;
    int x_sub = x + kDx[direction & 3];

    // 2. Initial safety check
    if (x_axis < 0 || x_axis >= config::Board::kWidth) {
        return;
    }
    if (x_sub < 0 || x_sub >= config::Board::kWidth) {
        // If sub is out of bounds, we might want to ignore the move or clip it.
        // Original Puyotan allows some kicks, but here we just ignore invalid moves for simplicity.
        return;
    }

    // 2. Place on spawn row (14th row)
    // In puyo, usually we place sub, then axis, or both.
    // If direction is vertical (0 or 2), they might share the same column.
    if (direction == 0) {
        // sub is above axis
        board_.placePiece(x_axis, piece.axis);
        Gravity::execute(board_); // drop axis
        board_.placePiece(x_sub,  piece.sub);
        Gravity::execute(board_); // drop sub
    } else if (direction == 2) {
        // axis is above sub
        board_.placePiece(x_sub,  piece.sub);
        Gravity::execute(board_); // drop sub
        board_.placePiece(x_axis, piece.axis);
        Gravity::execute(board_); // drop axis
    } else {
        // horizontal
        board_.placePiece(x_axis, piece.axis);
        board_.placePiece(x_sub,  piece.sub);
        Gravity::execute(board_);
    }

    // 3. Process chains
    // Optimization: only check colors of the piece in the first pass
    uint8_t color_mask = (1 << static_cast<int>(piece.axis)) | (1 << static_cast<int>(piece.sub));
    Chain::executeChain(board_, color_mask);

    // 4. Check death condition
    updateGameOver();

    // 5. Advance tsumo
    ++tsumo_index_;
}

void Simulator::updateGameOver() {
    // Check if the death cell is occupied
    if (board_.get(config::Rule::kDeathCol, config::Rule::kDeathRow) != Cell::Empty) {
        is_game_over_ = true;
    }
}

} // namespace puyotan
