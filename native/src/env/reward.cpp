#include <immintrin.h>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/env/reward.hpp>

namespace puyotan {
namespace {
// ---------------------------------------------------------------------------
// Board metric helpers
// ---------------------------------------------------------------------------

/**
 * Computes connectivity, isolation, near-group count, and color diversity.
 *
 * connectivity (out_conn):
 *   Count of puyos with >= 2 same-color neighbors.
 *
 * isolation (out_iso):
 *   Count of puyos with 0 same-color neighbors.
 *
 * near_group_count (out_near):
 *   Count of same-color groups of exactly 3 puyos (one away from firing).
 *   Detected via flood-fill on the BitBoard popcount of connected components.
 *   Approximated cheaply: detect puyos that have >= 2 same-color neighbors
 *   AND are adjacent to a puyo-pair (i.e., part of a 3+-cluster).
 *   Full exact count would require BFS; we use a fast bit-parallel approximation.
 *
 * color_diversity (out_div):
 *   Number of colors that have at least one connected group (conn > 0).
 */
void get_board_metrics(const Board& board,
                       int& out_conn, int& out_iso,
                       int& out_near, int& out_div) {
    out_conn = 0;
    out_iso = 0;
    out_near = 0;
    out_div = 0;

    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.empty())
            continue;

        const BitBoard U = bb.shiftUpRaw();
        const BitBoard D = bb.shiftDownRaw();
        const BitBoard L = bb.shiftLeftRaw();
        const BitBoard R = bb.shiftRightRaw();

        // Puyos with >= 2 same-color neighbors
        const BitBoard has_2 = bb & ((U & D) | (L & R) | ((U | D) & (L | R)));
        int conn = has_2.popcount();
        out_conn += conn;
        if (conn > 0)
            out_div++;

        // Isolated puyos (0 neighbors of same color)
        BitBoard iso = bb;
        iso.andNot(U | D | L | R);
        out_iso += iso.popcount();

        // Near-group approximation: puyos that have at least 2 same-color
        // neighbors AND at least one of those neighbors also has a neighbor.
        // = puyos in clusters of size >= 3.
        // Approximated as: (U|D|L|R) & has_2
        // i.e., neighbors of well-connected puyos.
        const BitBoard near_cluster = (U | D | L | R) & has_2;
        out_near += near_cluster.popcount();
    }
}

// ---------------------------------------------------------------------------
// Height variance
// ---------------------------------------------------------------------------
float get_height_variance(const Board& board) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int x = 0; x < config::Board::kWidth; ++x) {
        float h = static_cast<float>(board.getColumnHeight(x));
        sum += h;
        sum_sq += h * h;
    }
    constexpr float n = static_cast<float>(config::Board::kWidth);
    float mean = sum / n;
    return sum_sq / n - mean * mean;
}

// ---------------------------------------------------------------------------
// Buried puyo count (colored puyos under any ojama)
// ---------------------------------------------------------------------------
int get_buried_count(const Board& board) {
    const BitBoard& oj = board.getBitboard(Cell::Ojama);
    if (oj.empty())
        return 0;

    // Smear ojama downwards (shadow cast below each ojama)
    BitBoard s = oj;
    s |= s.shiftDownRaw();
    s |= _mm_srli_epi64(s.m128, 2);
    s |= _mm_srli_epi64(s.m128, 4);
    s |= _mm_srli_epi64(s.m128, 8);

    const BitBoard all_colored =
        board.getBitboard(Cell::Red) | board.getBitboard(Cell::Blue) |
        board.getBitboard(Cell::Green) | board.getBitboard(Cell::Yellow);
    return (all_colored & s).popcount();
}

// ---------------------------------------------------------------------------
// Potential chain (max chain achievable by adding exactly one puyo)
// ---------------------------------------------------------------------------
int get_max_potential_chain(const Board& board) {
    int max_chain = 0;
    for (int x = 0; x < config::Board::kWidth; ++x) {
        int h = board.getColumnHeight(x);
        if (h >= config::Board::kChainableRows)
            continue;

        for (int c = 0; c < config::Rule::kColors; ++c) {
            Board temp = board;
            temp.dropNewPiece(x, h, static_cast<Cell>(c));

            ErasureData ed = Chain::findGroups(temp, 1u << c);
            if (ed.num_erased > 0) {
                int chain = 0;
                while (ed.num_erased > 0) {
                    ++chain;
                    Chain::applyErasure(temp, ed);
                    uint32_t fallen_mask = Gravity::execute(temp);
                    ed = Chain::findGroups(temp, fallen_mask);
                }
                if (chain > max_chain)
                    max_chain = chain;
            }
        }
    }
    return max_chain;
}
} // namespace

// ---------------------------------------------------------------------------
// extractContext
// ---------------------------------------------------------------------------
RewardContext RewardCalculator::extractContext(const PuyotanMatch& m,
                                               const PuyotanPlayer& p1_pre, const PuyotanPlayer& p2_pre,
                                               int p1_ojama_dropped, int p2_ojama_dropped,
                                               int p1_max_chain, int p2_max_chain) const {
    RewardContext ctx;
    ctx.status = m.getStatus();

    auto get_colored_count = [](const Board& b) {
        return b.getBitboard(Cell::Red).popcount() +
               b.getBitboard(Cell::Green).popcount() +
               b.getBitboard(Cell::Blue).popcount() +
               b.getBitboard(Cell::Yellow).popcount();
    };

    // ---------------------------------------------------------------------------
    // Player 1
    // ---------------------------------------------------------------------------
    const auto& p1 = m.getPlayer(0);
    ctx.p1_delta_score = p1.score - p1_pre.score;
    ctx.p1_chain_count = p1_max_chain;
    
    // Calculate total erased via puyo count differences
    int p1_pre_colored = get_colored_count(p1_pre.field);
    int p1_post_colored = get_colored_count(p1.field);
    int p1_pre_ojama = p1_pre.field.getBitboard(Cell::Ojama).popcount();
    int p1_post_ojama = p1.field.getBitboard(Cell::Ojama).popcount();
    int p1_placed = (p1.active_next_pos > p1_pre.active_next_pos) ? 2 : 0;
    int p1_erased_colored = std::max(0, p1_pre_colored + p1_placed - p1_post_colored);
    int p1_erased_ojama = std::max(0, p1_pre_ojama + p1_ojama_dropped - p1_post_ojama);
    
    ctx.p1_total_erased = p1_erased_colored + p1_erased_ojama;
    ctx.p1_all_clear = (p1_max_chain > 0) && p1.field.getOccupied().empty();
    ctx.p1_ojama_sent = ctx.p1_delta_score / config::Score::kTargetScore;

    ctx.p1_puyo_count = p1.field.getOccupied().popcount();
    get_board_metrics(p1.field,
                      ctx.p1_connectivity_score,
                      ctx.p1_isolated_puyo_count,
                      ctx.p1_near_group_count,
                      ctx.p1_color_diversity);
    ctx.p1_height_variance = get_height_variance(p1.field);
    ctx.p1_death_col_height = p1.field.getColumnHeight(config::Rule::kDeathCol);
    ctx.p1_buried_puyo_count = get_buried_count(p1.field);
    ctx.p1_ojama_dropped = p1_ojama_dropped;
    ctx.p1_pending_ojama = p1.active_ojama + p1.non_active_ojama;
    ctx.p1_potential_chain = 0; // Disabled for performance (Phase 2 investigation)

    // ---------------------------------------------------------------------------
    // Player 2
    // ---------------------------------------------------------------------------
    const auto& p2 = m.getPlayer(1);
    ctx.p2_delta_score = p2.score - p2_pre.score;
    ctx.p2_chain_count = p2_max_chain;
    
    int p2_pre_colored = get_colored_count(p2_pre.field);
    int p2_post_colored = get_colored_count(p2.field);
    int p2_pre_ojama = p2_pre.field.getBitboard(Cell::Ojama).popcount();
    int p2_post_ojama = p2.field.getBitboard(Cell::Ojama).popcount();
    int p2_placed = (p2.active_next_pos > p2_pre.active_next_pos) ? 2 : 0;
    int p2_erased_colored = std::max(0, p2_pre_colored + p2_placed - p2_post_colored);
    int p2_erased_ojama = std::max(0, p2_pre_ojama + p2_ojama_dropped - p2_post_ojama);
    ctx.p2_total_erased = p2_erased_colored + p2_erased_ojama;
    ctx.p2_all_clear = (p2_max_chain > 0) && p2.field.getOccupied().empty();
    ctx.p2_ojama_sent = ctx.p2_delta_score / config::Score::kTargetScore;

    ctx.p2_puyo_count = p2.field.getOccupied().popcount();
    get_board_metrics(p2.field,
                      ctx.p2_connectivity_score,
                      ctx.p2_isolated_puyo_count,
                      ctx.p2_near_group_count,
                      ctx.p2_color_diversity);
    ctx.p2_height_variance = get_height_variance(p2.field);
    ctx.p2_death_col_height = p2.field.getColumnHeight(config::Rule::kDeathCol);
    ctx.p2_buried_puyo_count = get_buried_count(p2.field);
    ctx.p2_ojama_dropped = p2_ojama_dropped;
    ctx.p2_pending_ojama = p2.active_ojama + p2.non_active_ojama;
    ctx.p2_potential_chain = 0; // Disabled for performance

    return ctx;
}
} // namespace puyotan
