#include <puyotan/game/simulator.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/scorer.hpp>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace puyotan {

Simulator::Simulator(uint32_t seed) noexcept : tsumo_(seed) {}

void Simulator::reset(uint32_t seed) noexcept {
    board_ = Board();
    tsumo_.setSeed(seed);
    tsumo_index_ = 0;
    total_score_ = 0;
    is_game_over_ = false;
}

PuyoPiece Simulator::getCurrentPiece() noexcept {
    return tsumo_.get(tsumo_index_);
}

int Simulator::step(int x, Rotation rotation) noexcept {
    if (is_game_over_) return 0;

    const int score_before = total_score_;

    const PuyoPiece piece = getCurrentPiece();
    // Branchless wrap: kTsumoPoolSize is a power of 2, so AND mask eliminates branch
    tsumo_index_ = (tsumo_index_ + 1) & (config::Rule::kTsumoPoolSize - 1);

    const int r = std::to_underlying(rotation);
    const int x_axis = x;
    const int x_sub  = x_axis + kSubDx[r];

    // O(1) base height calculation
    const int h_axis = board_.getColumnHeight(x_axis);
    const int h_sub  = board_.getColumnHeight(x_sub);

    // kSoftDropBonusPerGrid == 1: multiply eliminated (static_assert in puyotan_match.cpp)
    total_score_ += (config::Board::kSpawnRow - std::max(h_axis, h_sub));

    // Direct BitBoard bit set (1 clock each, bypasses Gravity)
    board_.dropNewPiece(x_axis, h_axis + kAxisDy[r], piece.axis);
    board_.dropNewPiece(x_sub,  h_sub  + kSubDy_Simple[r], piece.sub);

    uint32_t dirty_colors = piece.dirty_flag;

    int chain_count = 0;
    while (dirty_colors & 0x0F) { // only normal colors chain
        ErasureData data = Chain::execute(board_, dirty_colors & 0x0F);
        if (data.num_erased == 0) break;

        ++chain_count;
        total_score_ += Scorer::calculateStepScore(data, chain_count);

        // All Clear check (Pure mathematical branchless)
        total_score_ += static_cast<int>(board_.getOccupied().empty()) * config::Score::kAllClearBonus;
        
        dirty_colors = Gravity::execute(board_);
    }

    updateGameOver();
    return total_score_ - score_before;
}

void Simulator::updateGameOver() noexcept {
    // Check the occupancy board once — faster than checking all 4 color planes.
    if (board_.getOccupied().get(config::Rule::kDeathCol, config::Rule::kDeathRow)) {
        is_game_over_ = true;
    }
}

int64_t Simulator::runBatch(int num_games, uint32_t seed) noexcept {
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

