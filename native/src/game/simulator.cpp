#include <puyotan/game/simulator.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>

namespace puyotan {

Simulator::Simulator(uint32_t seed) : tsumo_(seed) {}

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
    if (is_game_over_) return;

    PuyoPiece piece = getCurrentPiece();
    tsumo_index_++;

    // 1. Initial Gravity (Soft Drop) Calculation
    // We calculate distances BEFORE placement to avoid puyos blocking each other.
    // The bonus follows the puyo that falls the LESSER distance (Puyotan Beta rule).
    int sub_x = x;
    int sub_y = config::Board::kSpawnRow;
    switch (rotation) {
        case Rotation::Up:    sub_y += 1; break;
        case Rotation::Right: sub_x += 1; break;
        case Rotation::Down:  sub_y -= 1; break;
        case Rotation::Left:  sub_x -= 1; break;
    }

    int d1 = board_.getDropDistance(x, config::Board::kSpawnRow);
    int drop_dist = d1;
    if (sub_x >= 0 && sub_x < config::Board::kWidth) {
        int d2 = board_.getDropDistance(sub_x, sub_y);
        drop_dist = std::min(d1, d2);
    }

    // 2. Placement
    board_.placePiece(x, piece.axis);
    if (sub_x >= 0 && sub_x < config::Board::kWidth) {
        board_.set(sub_x, sub_y, piece.sub);
    }

    // 3. Execution & Scoring
    Gravity::execute(board_);
    total_score_ += std::max(0, drop_dist) * config::Score::kSoftDropBonusPerGrid;

    // 3. Chain Loop
    int chain_count = 0;
    while (true) {
        ErasureData data = Chain::execute(board_);
        if (!data.erased) break;

        chain_count++;
        total_score_ += Scorer::calculateStepScore(data, chain_count);
        Gravity::execute(board_);
    }

    updateGameOver();
}

void Simulator::updateGameOver() {
    if (board_.getBitboard(Cell::Red).get(config::Rule::kDeathCol, config::Rule::kDeathRow) ||
        board_.getBitboard(Cell::Green).get(config::Rule::kDeathCol, config::Rule::kDeathRow) ||
        board_.getBitboard(Cell::Blue).get(config::Rule::kDeathCol, config::Rule::kDeathRow) ||
        board_.getBitboard(Cell::Yellow).get(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
        is_game_over_ = true;
    }
}

} // namespace puyotan
