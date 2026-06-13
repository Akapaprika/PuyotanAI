/**
 * @file engine_benchmark.cpp
 * @brief Game Engine Performance Benchmark
 *
 * Measures:
 * - Frames per second (FPS): How many frames can be processed in 1 second
 * - Frames per game: Average frames until game ends
 * - Score per game: Average score achieved
 * - Moves per game: Average number of piece placements
 *
 * These metrics help verify that performance optimizations don't change game
 * behavior.
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/common/types.hpp>
#include <random>
#include <vector>

using namespace puyotan;

struct GameStats {
    int frames = 0;
    int score_p1 = 0;
    int score_p2 = 0;
    int moves_p1 = 0;
    int moves_p2 = 0;
    int chain_max_p1 = 0;
    int chain_max_p2 = 0;
    MatchStatus status = MatchStatus::Ready;
};

struct BenchmarkResult {
    double fps = 0.0;
    double games_per_sec = 0.0;
    uint64_t total_frames = 0;
    int total_games = 0;
    double elapsed_seconds = 0.0;
};

/// Simulates a single game with deterministic actions and returns statistics
GameStats simulateGame(uint32_t seed) {
    GameStats stats;
    PuyotanMatch match(seed);
    match.start();

    // Deterministic move pattern (same as runBatch for reproducibility)
    const int move_plan[] = {5, 5, 5, 5, 5, 5, 4, 4, 4,
                             4, 4, 4, 3, 3, 3, 3, 3, 3};
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);
    int p1_move = 0;
    int p2_move = 0;
    auto computePutColumn = [&](int move) noexcept {
        return move < num_moves ? move_plan[move] : 2;
    };

    while (match.getStatus() == MatchStatus::Playing) {
        bool action_set = false;
        int decision_mask = match.getDecisionMask();

        if (decision_mask & 1) {
            int col = computePutColumn(p1_move);
            if (match.setAction(0, Action{ActionType::Put,
                                          static_cast<int8_t>(col),
                                          Rotation::Up})) {
                ++p1_move;
                ++stats.moves_p1;
                action_set = true;
            }
        }

        if (decision_mask & 2) {
            int col = computePutColumn(p2_move);
            if (match.setAction(1, Action{ActionType::Put,
                                          static_cast<int8_t>(col),
                                          Rotation::Up})) {
                ++p2_move;
                ++stats.moves_p2;
                action_set = true;
            }
        }

        if (match.canStepNextFrame()) {
            match.stepNextFrame();
            ++stats.frames;
        } else if (!action_set) {
            // Stuck state - should not happen
            break;
        }
    }

    // Record final stats
    stats.status = match.getStatus();
    stats.score_p1 = match.getPlayer(0).score;
    stats.score_p2 = match.getPlayer(1).score;
    stats.chain_max_p1 = match.getPlayer(0).chain_count;
    stats.chain_max_p2 = match.getPlayer(1).chain_count;

    return stats;
}

/// Runs benchmark for specified duration and returns results
BenchmarkResult runBenchmark(double duration_seconds, uint32_t base_seed = 1) {
    BenchmarkResult result;

    auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t seed = base_seed;
    int check_counter = 0;

    while (true) {
        // Query clock only once every 1000 games to minimize clock-call
        // overhead
        if (++check_counter >= 100000) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed =
                std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= duration_seconds)
                break;
            check_counter = 0;
        }

        GameStats stats = simulateGame(seed++);
        result.total_frames += stats.frames;
        ++result.total_games;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    result.fps = result.total_frames / result.elapsed_seconds;
    result.games_per_sec = result.total_games / result.elapsed_seconds;

    return result;
}

/// Prints benchmark results in a readable format
void printBenchmarkResult(const BenchmarkResult& r) {
    printf("\n========================================\n");
    printf("  ENGINE PERFORMANCE BENCHMARK RESULTS\n");
    printf("========================================\n");
    printf("Duration:        %.3f seconds\n", r.elapsed_seconds);
    printf("Total Games:     %d\n", r.total_games);
    printf("Total Frames:    %llu\n", r.total_frames);
    printf("\n--- Throughput ---\n");
    printf("FPS (frames/s):  %.0f\n", r.fps);
    printf("Games/s:         %.2f\n", r.games_per_sec);
    printf("Usec/frame:      %.3f\n", 1000000.0 / r.fps);
    printf("Avg Frames/game: %.1f\n",
           static_cast<double>(r.total_frames) / r.total_games);
    printf("========================================\n");
}

struct ExpectedEngineStats {
    uint32_t seed;
    int frames;
    int score_p1;
    int score_p2;
    int moves_p1;
    int moves_p2;
    int status;
};

/// Quick verification: runs a few games with fixed seeds and prints stats for
/// regression testing. Returns true if all stats match expected values.
bool runRegressionTest() {
    printf("\n=== REGRESSION TEST (Fixed Seeds) ===\n");
    const ExpectedEngineStats expected[] = {
        {1, 56, 286, 286, 25, 25, 4},
        {42, 51, 300, 300, 24, 24, 4},
        {123, 61, 816, 816, 26, 26, 4},
        {999, 66, 1390, 1390, 26, 26, 4},
        {12345, 56, 280, 280, 25, 25, 4},
        {424242, 55, 319, 319, 24, 24, 4}
    };

    bool all_ok = true;
    for (const auto& exp : expected) {
        GameStats s = simulateGame(exp.seed);
        printf("Seed %7u: frames=%4d  score=(%4d,%4d)  moves=(%3d,%3d)  "
               "max_chain=(%d,%d)  status=%d\n",
               exp.seed, s.frames, s.score_p1, s.score_p2, s.moves_p1, s.moves_p2,
               s.chain_max_p1, s.chain_max_p2, static_cast<int>(s.status));
        
        if (s.frames != exp.frames || s.score_p1 != exp.score_p1 || s.score_p2 != exp.score_p2 ||
            s.moves_p1 != exp.moves_p1 || s.moves_p2 != exp.moves_p2 || static_cast<int>(s.status) != exp.status) {
            printf("  [ERROR] Regression mismatch for Seed %u!\n", exp.seed);
            printf("          Expected: frames=%d, score=(%d,%d), moves=(%d,%d), status=%d\n",
                   exp.frames, exp.score_p1, exp.score_p2, exp.moves_p1, exp.moves_p2, exp.status);
            all_ok = false;
        }
    }
    printf("======================================\n\n");
    return all_ok;
}

int main(int argc, char** argv) {
    double duration = 5.0; // Default 5 seconds
    bool run_regression = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" || arg == "-d") {
            if (i + 1 < argc)
                duration = std::stod(argv[++i]);
        } else if (arg == "--regression" || arg == "-r") {
            run_regression = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: engine_benchmark [options]\n");
            printf("Options:\n");
            printf("  --duration, -d <seconds>  Benchmark duration (default: 5.0)\n");
            printf("  --regression, -r          Run regression test with fixed seeds\n");
            printf("  --help, -h                Show this help\n");
            return 0;
        }
    }

    printf("Engine Benchmark Starting...\n");
    printf("Duration: %.1f seconds\n", duration);

    if (run_regression) {
        bool ok = runRegressionTest();
        if (!ok) {
            printf("Regression test FAILED!\n");
            return 1;
        }
        printf("Regression test PASSED!\n");
        return 0;
    }

    BenchmarkResult result = runBenchmark(duration);
    printBenchmarkResult(result);

    bool ok = runRegressionTest();
    if (!ok) {
        printf("Post-benchmark regression check FAILED!\n");
        return 1;
    }

    return 0;
}