#include <puyotan/game/simulator.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace puyotan {

Simulator::Simulator(int32_t seed) : tsumo_(seed) {}

void Simulator::reset(int32_t seed) {
    board_ = Board();
    tsumo_.setSeed(seed);
    tsumo_index_ = 0;
    total_score_ = 0;
    is_game_over_ = false;
}

PuyoPiece Simulator::getCurrentPiece() const {
    return tsumo_.get(tsumo_index_);
}

int Simulator::step(int x, Rotation rotation) {
    if (is_game_over_) return 0;

    const int score_before = total_score_;

    PuyoPiece piece = getCurrentPiece();
    if (++tsumo_index_ >= config::Rule::kTsumoPoolSize) {
        tsumo_index_ = 0;
    }

    // -----------------------------------------------------------------------
    // Branchless placement via compile-time LUT indexed by Rotation (0-3).
    //   Up=0:    axis at (x, h),     sub at (x,   h+1)
    //   Right=1: axis at (x, h_ax),  sub at (x+1, h_sub)
    //   Down=2:  axis at (x, h+1),   sub at (x,   h)
    //   Left=3:  axis at (x, h_ax),  sub at (x-1, h_sub)
    // -----------------------------------------------------------------------
    const int r = std::to_underlying(rotation);
    const int h_axis = board_.getColumnHeight(x);

    const int sub_x  = x + kSubDx[r];
    assert(sub_x >= 0 && sub_x < config::Board::kWidth);

    const int h_sub = board_.getColumnHeight(sub_x);

    const int final_y_axis = h_axis + kAxisDy[r];
    const int final_y_sub  = h_sub  + kSubDy_Simple[r];

    const int drop_dist   = config::Board::kSpawnRow - std::max(h_axis, h_sub);
    total_score_ += drop_dist * config::Score::kSoftDropBonusPerGrid;

    board_.dropNewPiece(x, final_y_axis, piece.axis);
    board_.dropNewPiece(sub_x, final_y_sub, piece.sub);

    uint32_t dirty_colors = (1u << std::to_underlying(piece.axis)) | (1u << std::to_underlying(piece.sub));

    int chain_count = 0;
    while (dirty_colors & 0x0F) { // only normal colors chain
        ErasureData data = Chain::execute(board_, dirty_colors & 0x0F);
        if (data.num_erased == 0) break;

        ++chain_count;
        total_score_ += Scorer::calculateStepScore(data, chain_count);
        
        dirty_colors = Gravity::execute(board_);
    }

    updateGameOver();
    return total_score_ - score_before;
}

void Simulator::updateGameOver() {
    // Check the occupancy board once — faster than checking all 4 color planes.
    if (board_.getOccupied().get(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
        is_game_over_ = true;
    }
}

int64_t Simulator::runBatch(int num_games, int32_t seed) {
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

