#include <puyotan/game/simulator.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>
#include <cstdint>

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
    if (++tsumo_index_ >= config::Rule::kTsumoPoolSize) {
        tsumo_index_ = 0;
    }

    int h_axis = board_.getColumnHeight(x);
    int final_y_axis, final_y_sub;
    int sub_x = x;
    int drop_dist;

    if (rotation == Rotation::Up) {
        final_y_axis = h_axis;
        final_y_sub = h_axis + 1;
        drop_dist = config::Board::kSpawnRow - final_y_axis;
    } else if (rotation == Rotation::Down) {
        final_y_sub = h_axis;
        final_y_axis = h_axis + 1;
        drop_dist = config::Board::kSpawnRow - final_y_axis;
    } else {
        sub_x = (rotation == Rotation::Right) ? (x + 1) : (x - 1);
        if (sub_x >= 0 && sub_x < config::Board::kWidth) {
            int h_sub = board_.getColumnHeight(sub_x);
            final_y_axis = h_axis;
            final_y_sub = h_sub;
            drop_dist = std::min(config::Board::kSpawnRow - final_y_axis, 
                                 config::Board::kSpawnRow - final_y_sub);
        } else {
            final_y_axis = h_axis;
            final_y_sub = -1;
            drop_dist = config::Board::kSpawnRow - final_y_axis;
        }
    }

    board_.dropNewPiece(x, final_y_axis, piece.axis);
    if (final_y_sub >= 0) {
        board_.dropNewPiece(sub_x, final_y_sub, piece.sub);
    }

    total_score_ += std::max(0, drop_dist) * config::Score::kSoftDropBonusPerGrid;

    uint8_t dirty_colors = 0;
    if (piece.axis != Cell::Empty && piece.axis != Cell::Ojama) {
        dirty_colors |= (1 << static_cast<int>(piece.axis));
    }
    if (piece.sub != Cell::Empty && piece.sub != Cell::Ojama) {
        dirty_colors |= (1 << static_cast<int>(piece.sub));
    }

    int chain_count = 0;
    while (dirty_colors & 0x0F) { // only normal colors chain
        ErasureData data = Chain::execute(board_, dirty_colors & 0x0F);
        if (!data.erased) break;

        ++chain_count;
        total_score_ += Scorer::calculateStepScore(data, chain_count);
        
        if (Gravity::canFall(board_)) {
            dirty_colors = Gravity::execute(board_);
        } else {
            dirty_colors = 0;
        }
    }

    updateGameOver();
}

void Simulator::updateGameOver() {
    // Check the occupancy board once — faster than checking all 4 color planes.
    if (board_.getOccupied().get(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
        is_game_over_ = true;
    }
}

int64_t Simulator::runBatch(int num_games, uint32_t seed) {
    // Move pattern matching tests/benchmark.py:
    //   6 moves at col 5, 6 at col 4, 6 at col 3, then col 2 until game over.
    int64_t total_steps = 0;
    for (int i = 0; i < num_games; ++i) {
        reset(seed);

        // Phase 1: col 5 × 6
        for (int j = 0; j < 6 && !is_game_over_; ++j) {
            step(5, Rotation::Up);
            ++total_steps;
        }
        // Phase 2: col 4 × 6
        for (int j = 0; j < 6 && !is_game_over_; ++j) {
            step(4, Rotation::Up);
            ++total_steps;
        }
        // Phase 3: col 3 × 6
        for (int j = 0; j < 6 && !is_game_over_; ++j) {
            step(3, Rotation::Up);
            ++total_steps;
        }
        // Phase 4: col 2 until game over
        while (!is_game_over_) {
            step(2, Rotation::Up);
            ++total_steps;
        }
    }
    return total_steps;
}

} // namespace puyotan

