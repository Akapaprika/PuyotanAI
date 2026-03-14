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
    total_score_ = 0;
    is_game_over_ = false;
}

PuyoPiece Simulator::getCurrentPiece() const {
    return tsumo_.get(tsumo_index_);
}

void Simulator::step(int x, Rotation rotation) {
    if (is_game_over_) {
        return;
    }

    PuyoPiece piece = getCurrentPiece();

    // 1. Determine spawn positions
    // Axis is always at x.
    // Sub direction: 0:up, 1:right, 2:down, 3:left
    static constexpr int kDx[] = {0, 1, 0, -1};
    int x_axis = x;
    int x_sub = x + kDx[static_cast<int>(rotation) & 3];

    // 2. Initial safety check
    if (x_axis < 0 || x_axis >= config::Board::kWidth) {
        return;
    }
    if (x_sub < 0 || x_sub >= config::Board::kWidth) {
        return;
    }

    int drop_dist = 0;

    // 2. Place on spawn row (14th row)
    if (rotation == Rotation::Up) {
        // sub is above axis
        board_.placePiece(x_axis, piece.axis);
        int d1 = Gravity::execute(board_); // drop axis
        board_.placePiece(x_sub,  piece.sub);
        int d2 = Gravity::execute(board_); // drop sub
        // piece drop distance is the maximum distance any of its parts fell
        drop_dist = std::max(d1, d2);
    } else if (rotation == Rotation::Down) {
        // axis is above sub
        board_.placePiece(x_sub,  piece.sub);
        int d1 = Gravity::execute(board_); // drop sub
        board_.placePiece(x_axis, piece.axis);
        int d2 = Gravity::execute(board_); // drop axis
        drop_dist = std::max(d1, d2);
    } else {
        // horizontal
        board_.placePiece(x_axis, piece.axis);
        board_.placePiece(x_sub,  piece.sub);
        drop_dist = Gravity::execute(board_);
    }

    // Add soft drop bonus
    total_score_ += drop_dist * config::Score::kSoftDropBonusPerGrid;

    // 3. Process chains
    uint8_t color_mask = (1 << static_cast<int>(piece.axis)) | (1 << static_cast<int>(piece.sub));
    Chain::ChainResult result = Chain::executeChain(board_, color_mask);
    total_score_ += result.total_score;

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
