#include <puyotan/game/simulator.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>
#include <cstdint>
#include <utility>

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

    // -----------------------------------------------------------------------
    // Branchless placement via compile-time LUT indexed by Rotation (0-3).
    //   Up=0:    axis at (x, h),     sub at (x,   h+1)
    //   Right=1: axis at (x, h_ax),  sub at (x+1, h_sub)
    //   Down=2:  axis at (x, h+1),   sub at (x,   h)
    //   Left=3:  axis at (x, h_ax),  sub at (x-1, h_sub)
    // -----------------------------------------------------------------------
    static constexpr int8_t kAxisDy[4] = { 0,  0,  1,  0 }; // Down: axis 1 above floor
    static constexpr int8_t kSubDx[4]  = { 0,  1,  0, -1 }; // Right/Left: sub is in adjacent col
    static constexpr int8_t kSubDy[4]  = { 1,  0, -1,  0 }; // Up/Down: sub relative to axis y

    const int r = std::to_underlying(rotation);
    const int h_axis = board_.getColumnHeight(x);

    const int sub_dx   = kSubDx[r];
    const int sub_x    = x + sub_dx;
    // For horizontal, sub needs its own column height; clamp to avoid OOB on height query
    const int sub_x_safe = sub_dx ? std::clamp(sub_x, 0, config::Board::kWidth - 1) : x;
    const int h_sub    = board_.getColumnHeight(sub_x_safe);  // free query, branchless

    const int final_y_axis = h_axis + kAxisDy[r];
    // Vertical: sub y = final_y_axis + kSubDy[r]. Horizontal: sub y = h_sub (if in bounds).
    const bool is_horiz     = (sub_dx != 0);
    const bool sub_in_range = (sub_x >= 0) & (sub_x < config::Board::kWidth);
    const int  final_y_sub  = is_horiz
        ? (sub_in_range ? h_sub : -1)
        : (final_y_axis + kSubDy[r]);

    // drop_dist: min fall among both pieces.
    const int drop_dist = is_horiz && sub_in_range
        ? (config::Board::kSpawnRow - std::max(h_axis, h_sub))
        : (config::Board::kSpawnRow - final_y_axis);

    board_.dropNewPiece(x, final_y_axis, piece.axis);
    if (final_y_sub >= 0) {
        board_.dropNewPiece(sub_x, final_y_sub, piece.sub);
    }

    total_score_ += std::max(0, drop_dist) * config::Score::kSoftDropBonusPerGrid;

    uint8_t dirty_colors = 0;
    if (piece.axis != Cell::Empty && piece.axis != Cell::Ojama) {
        dirty_colors |= (1 << std::to_underlying(piece.axis));
    }
    if (piece.sub != Cell::Empty && piece.sub != Cell::Ojama) {
        dirty_colors |= (1 << std::to_underlying(piece.sub));
    }

    int chain_count = 0;
    while (dirty_colors & 0x0F) { // only normal colors chain
        ErasureData data = Chain::execute(board_, dirty_colors & 0x0F);
        if (!data.erased) break;

        ++chain_count;
        total_score_ += Scorer::calculateStepScore(data, chain_count);
        
        dirty_colors = Gravity::execute(board_);
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

