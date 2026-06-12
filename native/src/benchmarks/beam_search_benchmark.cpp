/**
 * @file beam_search_benchmark.cpp
 * @brief Beam Search Performance Benchmark (Solo Mode Simulation)
 *
 * Measures:
 * - Searches per second: How many beam search operations can be completed in 1
 * second
 * - Nodes per second: Total beam nodes evaluated per second
 * - FPS (Frames per second): Overall match simulation speed
 * - Placements per second: Dynamic decision-making throughput (Moves per
 * second)
 * - Latency: Time per search (p50, p95, p99) under realistic solo workloads
 * - Action distribution: Which actions are selected in actual play
 * - Score statistics: Expected scores from search
 *
 * Solo mode is simulated by mirroring the exact same actions on both 1P and 2P.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/search/beam_search.hpp>
#include <random>
#include <string>
#include <vector>

using namespace puyotan;
using namespace puyotan::search;

struct SearchStats {
    int action = -1;
    float expected_score = 0.0f;
    double latency_ms = 0.0;
    bool valid = false;
};

struct BenchmarkResult {
    double searches_per_sec = 0.0;
    double nodes_per_sec = 0.0;
    double fps = 0.0;
    double moves_per_sec = 0.0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    SearchStats avg_stats{};
    std::array<int, kNumRLActions> action_counts{};
    uint64_t total_searches = 0;
    uint64_t total_nodes = 0;
    uint64_t total_frames = 0;
    uint64_t total_moves = 0;
    int total_games = 0;
    double elapsed_seconds = 0.0;
};

/// Estimates nodes processed in a beam search (approximate)
int estimateNodesProcessed(const BeamConfig& cfg) {
    const int actions_per_depth =
        18; // ~18 put actions (excluding duplicates and pass)
    int nodes = 0;
    int current_width = 1;
    for (int d = 0; d < cfg.look_ahead; ++d) {
        int expanded = current_width * actions_per_depth;
        nodes += expanded;
        current_width = std::min(expanded, cfg.beam_width);
    }
    if (cfg.look_ahead >= 4) {
        nodes += current_width * 10 * actions_per_depth; // 10 tsumo patterns
        if (cfg.look_ahead >= 5) {
            nodes += current_width * 10 * 10 *
                     actions_per_depth; // 2-step expectimax
        }
    }
    return nodes;
}

/// Creates a realistic player state for regression testing (10 preliminary
/// moves)
PuyotanPlayer createTestPlayer(uint32_t seed) {
    PuyotanPlayer player;
    PuyotanMatch match(seed);
    match.start();

    const int move_plan[] = {2, 3, 2, 3, 4, 3, 2, 4, 3, 2};
    int move_idx = 0;
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);

    for (int i = 0; i < 20 && match.getStatus() == MatchStatus::Playing; ++i) {
        if (match.getPlayer(0).current_action.action.type == ActionType::None) {
            int col = move_plan[move_idx % num_moves];
            match.setAction(0, Action{ActionType::Put, static_cast<int8_t>(col),
                                      Rotation::Up});
            ++move_idx;
        }
        if (match.canStepNextFrame()) {
            match.stepNextFrame();
        }
    }

    player = match.getPlayer(0);
    return player;
}

/// Runs a single beam search and returns statistics
SearchStats runSingleSearch(const PuyotanPlayer& player, const Tsumo& tsumo,
                            const BeamConfig& cfg) {
    SearchStats stats;
    auto start = std::chrono::high_resolution_clock::now();

    auto result = beamSearch(player, tsumo, cfg);

    auto end = std::chrono::high_resolution_clock::now();
    stats.latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    stats.action = result.first;
    stats.expected_score = result.second;
    stats.valid = (result.first >= 0 && result.first < kNumRLActions);

    return stats;
}

/// Runs benchmark in simulated solo format
BenchmarkResult runBenchmark(double duration_seconds, const BeamConfig& cfg,
                             uint32_t base_seed = 12345) {
    BenchmarkResult result;
    std::vector<double> latencies;
    latencies.reserve(200000);
    std::vector<float> expected_scores;
    expected_scores.reserve(200000);

    auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t seed = base_seed;

    // Safety limits to prevent out-of-bounds tsumo access and infinite loops
    const int max_moves_per_game =
        100; // Cap at 100 placements to stay safely within Tsumo bounds

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= duration_seconds)
            break;

        PuyotanMatch match(seed);
        Tsumo tsumo(seed);
        match.start();

        int game_moves = 0;

        // Match progression loop
        while (match.getStatus() == MatchStatus::Playing &&
               game_moves < max_moves_per_game) {
            // Periodical duration timeout check
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_current =
                std::chrono::duration<double>(current_time - start_time)
                    .count();
            if (elapsed_current >= duration_seconds) {
                break;
            }

            bool action_set = false;
            int decision_mask = match.getDecisionMask();

            // Run beam search once at Player 0 (1P) decision timing
            if (decision_mask & 1) {
                auto start = std::chrono::high_resolution_clock::now();
                auto search_res = beamSearch(match.getPlayer(0), tsumo, cfg);
                auto end = std::chrono::high_resolution_clock::now();

                double latency_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
                latencies.push_back(latency_ms);
                expected_scores.push_back(search_res.second);

                int action_idx = search_res.first;
                if (action_idx >= 0 && action_idx < kNumRLActions) {
                    Action act = getRLAction(action_idx);
                    result.action_counts[action_idx]++;

                    // Apply identical action to both 1P and 2P to maintain
                    // perfect symmetry (Solo Mode)
                    match.setAction(0, act);
                    match.setAction(1, act);

                    result.total_moves++;
                    game_moves++;
                    action_set = true;
                } else {
                    // No valid action found (game over/dead end) -> exit match
                    // loop immediately
                    break;
                }
                result.total_nodes += estimateNodesProcessed(cfg);
                result.total_searches++;
            }

            // Advance match frames
            if (match.canStepNextFrame()) {
                match.stepNextFrame();
                result.total_frames++;
            } else if (!action_set) {
                // Deadlock prevention
                break;
            }
        }
        result.total_games++;
        seed++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    // Throughput metrics
    result.searches_per_sec = result.total_searches / result.elapsed_seconds;
    result.nodes_per_sec = result.total_nodes / result.elapsed_seconds;
    result.fps = result.total_frames / result.elapsed_seconds;
    result.moves_per_sec = result.total_moves / result.elapsed_seconds;

    // Compute latency percentiles
    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        size_t n = latencies.size();
        result.p50_latency_ms = latencies[n * 50 / 100];
        result.p95_latency_ms = latencies[n * 95 / 100];
        result.p99_latency_ms = latencies[n * 99 / 100];

        double sum_latency =
            std::accumulate(latencies.begin(), latencies.end(), 0.0);
        result.avg_latency_ms = sum_latency / n;
    }

    if (!expected_scores.empty()) {
        float sum_scores = std::accumulate(expected_scores.begin(),
                                           expected_scores.end(), 0.0f);
        result.avg_stats.expected_score = sum_scores / expected_scores.size();
        result.avg_stats.valid = true;
    }

    return result;
}

void printBenchmarkResult(const BenchmarkResult& r, const BeamConfig& cfg) {
    printf("\n========================================\n");
    printf("  BEAM SEARCH SOLO-MODE BENCHMARK\n");
    printf("========================================\n");
    printf("Config: beam_width=%d, look_ahead=%d, fast_potential=%d\n",
           cfg.beam_width, cfg.look_ahead, cfg.eval_weights.use_fast_potential);
    printf("Duration:        %.3f seconds\n", r.elapsed_seconds);
    printf("Total Games:     %d\n", r.total_games);
    printf("Total Frames:    %llu\n", r.total_frames);
    printf("Total Placements:%llu\n", r.total_moves);
    printf("Total Searches:  %llu\n", r.total_searches);
    printf("Est. Total Nodes:%llu\n", r.total_nodes);
    printf("\n--- Throughput ---\n");
    printf("FPS (frames/s):  %.0f\n", r.fps);
    printf("Placements/sec:  %.2f (Moves/s)\n", r.moves_per_sec);
    printf("Searches/sec:    %.0f\n", r.searches_per_sec);
    printf("Nodes/sec:       %.0f (%.2f M)\n", r.nodes_per_sec,
           r.nodes_per_sec / 1e6);
    printf("\n--- Latency (Under Solo-Game Workload) ---\n");
    printf("Avg:             %.3f ms\n", r.avg_latency_ms);
    printf("P50:             %.3f ms\n", r.p50_latency_ms);
    printf("P95:             %.3f ms\n", r.p95_latency_ms);
    printf("P99:             %.3f ms (Worst-case scenario)\n",
           r.p99_latency_ms);
    printf("\n--- Search Quality ---\n");
    printf("Avg Expected Score: %.2f\n", r.avg_stats.expected_score);
    printf("\n--- Action Distribution ---\n");
    for (int i = 0; i < kNumRLActions; ++i) {
        if (r.action_counts[i] > 0) {
            Action a = getRLAction(i);
            if (a.type == ActionType::Put) {
                double pct = 100.0 * r.action_counts[i] / r.total_searches;
                printf("  Action %3d (Put, rot=%d, x=%2d): %6d (%.1f%%)\n", i,
                       static_cast<int>(a.rotation), a.x, r.action_counts[i],
                       pct);
            }
        }
    }
    printf("========================================\n");
}

void runRegressionTest(const BeamConfig& cfg) {
    printf("\n=== REGRESSION TEST (Fixed Seeds) ===\n");
    const uint32_t test_seeds[] = {1,     42,     123,    999,
                                   12345, 424242, 111111, 999999};

    for (uint32_t seed : test_seeds) {
        PuyotanPlayer player = createTestPlayer(seed);
        Tsumo tsumo(seed);
        SearchStats stats = runSingleSearch(player, tsumo, cfg);
        Action a = getRLAction(stats.action);
        printf("Seed %7u: action=%3d (Put, rot=%d, x=%2d)  score=%.2f  "
               "latency=%.3fms  valid=%d\n",
               seed, stats.action, static_cast<int>(a.rotation), a.x,
               stats.expected_score, stats.latency_ms, stats.valid);
    }
    printf("======================================\n\n");
}

int main(int argc, char** argv) {
    double duration = 5.0;
    bool run_regression = false;
    int beam_width = 500;
    int look_ahead = 3;
    bool fast_potential = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" || arg == "-d") {
            if (i + 1 < argc)
                duration = std::stod(argv[++i]);
        } else if (arg == "--regression" || arg == "-r") {
            run_regression = true;
        } else if (arg == "--beam-width" || arg == "-w") {
            if (i + 1 < argc)
                beam_width = std::stoi(argv[++i]);
        } else if (arg == "--look-ahead" || arg == "-l") {
            if (i + 1 < argc)
                look_ahead = std::stoi(argv[++i]);
        } else if (arg == "--fast-potential" || arg == "-f") {
            fast_potential = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: beam_search_benchmark [options]\n");
            printf("Options:\n");
            printf("  --duration, -d <seconds>     Benchmark duration "
                   "(default: 5.0)\n");
            printf("  --regression, -r             Run regression test with "
                   "fixed seeds\n");
            printf(
                "  --beam-width, -w <int>       Beam width (default: 500)\n");
            printf("  --look-ahead, -l <int>       Look ahead depth (default: "
                   "3)\n");
            printf("  --fast-potential, -f         Use fast potential "
                   "evaluation\n");
            printf("  --help, -h                   Show this help\n");
            return 0;
        }
    }

    BeamConfig cfg;
    cfg.beam_width = beam_width;
    cfg.look_ahead = look_ahead;
    cfg.eval_weights.use_fast_potential = fast_potential;

    printf("Beam Search Benchmark (Solo-Mode format) Starting...\n");
    printf("Duration: %.1f seconds\n", duration);
    printf("Config: width=%d, depth=%d, fast_potential=%d\n", beam_width,
           look_ahead, fast_potential);

    if (run_regression) {
        runRegressionTest(cfg);
    }

    BenchmarkResult result = runBenchmark(duration, cfg);
    printBenchmarkResult(result, cfg);

    return 0;
}