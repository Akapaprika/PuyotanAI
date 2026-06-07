#include <immintrin.h>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/scorer.hpp>
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
std::pair<int, int> get_max_potential_chain_and_score(const Board& board) {
    int max_chain = 0;
    int max_score = 0;
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
                int score = 0;
                while (ed.num_erased > 0) {
                    ++chain;
                    score += Scorer::calculateStepScore(ed, chain);
                    Chain::applyErasure(temp, ed);
                    uint32_t fallen_mask = Gravity::execute(temp);
                    ed = Chain::findGroups(temp, fallen_mask);
                    if (chain >= 10) {
                        return {chain, score};
                    }
                }
                if (chain > max_chain) {
                    max_chain = chain;
                }
                if (score > max_score) {
                    max_score = score;
                }
            }
        }
    }
    return {max_chain, max_score};
}
struct SinglePlayerRewardContext {
    int delta_score = 0;
    int chain_count = 0;
    int total_erased = 0;
    bool all_clear = false;
    int ojama_sent = 0;
    int puyo_count = 0;
    int connectivity_score = 0;
    int isolated_puyo_count = 0;
    int near_group_count = 0;
    int color_diversity = 0;
    float height_variance = 0.0f;
    int death_col_height = 0;
    int buried_puyo_count = 0;
    int ojama_dropped = 0;
    int pending_ojama = 0;
    int potential_chain = 0;
    int potential_score = 0;
};

SinglePlayerRewardContext extractSinglePlayerMetrics(const PuyotanPlayer& p, const PuyotanPlayer& p_pre,
                                                     int ojama_dropped, int max_chain) {
    SinglePlayerRewardContext sp;
    sp.delta_score = p.score - p_pre.score;
    sp.chain_count = max_chain;

    auto get_colored_count = [](const Board& b) {
        return b.getBitboard(Cell::Red).popcount() +
               b.getBitboard(Cell::Green).popcount() +
               b.getBitboard(Cell::Blue).popcount() +
               b.getBitboard(Cell::Yellow).popcount();
    };

    int pre_colored = get_colored_count(p_pre.field);
    int post_colored = get_colored_count(p.field);
    int pre_ojama = p_pre.field.getBitboard(Cell::Ojama).popcount();
    int post_ojama = p.field.getBitboard(Cell::Ojama).popcount();
    int placed = (p.active_next_pos > p_pre.active_next_pos) ? 2 : 0;
    int erased_colored = std::max(0, pre_colored + placed - post_colored);
    int erased_ojama = std::max(0, pre_ojama + ojama_dropped - post_ojama);

    sp.total_erased = erased_colored + erased_ojama;
    sp.all_clear = (max_chain > 0) && p.field.getOccupied().empty();
    sp.ojama_sent = sp.delta_score / config::Score::kTargetScore;
    sp.puyo_count = p.field.getOccupied().popcount();

    get_board_metrics(p.field,
                      sp.connectivity_score,
                      sp.isolated_puyo_count,
                      sp.near_group_count,
                      sp.color_diversity);

    sp.height_variance = get_height_variance(p.field);
    sp.death_col_height = p.field.getColumnHeight(config::Rule::kDeathCol);
    sp.buried_puyo_count = get_buried_count(p.field);
    sp.ojama_dropped = ojama_dropped;
    sp.pending_ojama = p.active_ojama + p.non_active_ojama;

    auto [pot_chain, pot_score] = get_max_potential_chain_and_score(p.field);
    sp.potential_chain = pot_chain;
    sp.potential_score = pot_score;

    return sp;
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

    // ---------------------------------------------------------------------------
    // Player 1
    // ---------------------------------------------------------------------------
    const auto& p1 = m.getPlayer(0);
    SinglePlayerRewardContext sp1 = extractSinglePlayerMetrics(p1, p1_pre, p1_ojama_dropped, p1_max_chain);

    ctx.p1_delta_score         = sp1.delta_score;
    ctx.p1_chain_count         = sp1.chain_count;
    ctx.p1_total_erased        = sp1.total_erased;
    ctx.p1_all_clear           = sp1.all_clear;
    ctx.p1_ojama_sent          = sp1.ojama_sent;
    ctx.p1_puyo_count          = sp1.puyo_count;
    ctx.p1_connectivity_score  = sp1.connectivity_score;
    ctx.p1_isolated_puyo_count = sp1.isolated_puyo_count;
    ctx.p1_near_group_count    = sp1.near_group_count;
    ctx.p1_color_diversity     = sp1.color_diversity;
    ctx.p1_height_variance     = sp1.height_variance;
    ctx.p1_death_col_height    = sp1.death_col_height;
    ctx.p1_buried_puyo_count   = sp1.buried_puyo_count;
    ctx.p1_ojama_dropped       = sp1.ojama_dropped;
    ctx.p1_pending_ojama       = sp1.pending_ojama;
    ctx.p1_potential_chain     = sp1.potential_chain;
    ctx.p1_potential_score     = sp1.potential_score;

    // ---------------------------------------------------------------------------
    // Player 2
    // ---------------------------------------------------------------------------
    const auto& p2 = m.getPlayer(1);
    if (skip_opponent_metrics_) {
        // Skip expensive opponent metrics when opponent rewards are all zero.
        ctx.p2_delta_score         = p2.score - p2_pre.score;
        ctx.p2_chain_count         = p2_max_chain;
        ctx.p2_total_erased        = 0;
        ctx.p2_all_clear           = false;
        ctx.p2_ojama_sent          = ctx.p2_delta_score / config::Score::kTargetScore;
        ctx.p2_puyo_count          = p2.field.getOccupied().popcount();
        ctx.p2_connectivity_score  = 0;
        ctx.p2_isolated_puyo_count = 0;
        ctx.p2_near_group_count    = 0;
        ctx.p2_color_diversity     = 0;
        ctx.p2_height_variance     = 0.0f;
        ctx.p2_death_col_height    = 0;
        ctx.p2_buried_puyo_count   = 0;
        ctx.p2_ojama_dropped       = p2_ojama_dropped;
        ctx.p2_pending_ojama       = p2.active_ojama + p2.non_active_ojama;
        ctx.p2_potential_chain     = 0;
        ctx.p2_potential_score     = 0;
    } else {
        SinglePlayerRewardContext sp2 = extractSinglePlayerMetrics(p2, p2_pre, p2_ojama_dropped, p2_max_chain);
        ctx.p2_delta_score         = sp2.delta_score;
        ctx.p2_chain_count         = sp2.chain_count;
        ctx.p2_total_erased        = sp2.total_erased;
        ctx.p2_all_clear           = sp2.all_clear;
        ctx.p2_ojama_sent          = sp2.ojama_sent;
        ctx.p2_puyo_count          = sp2.puyo_count;
        ctx.p2_connectivity_score  = sp2.connectivity_score;
        ctx.p2_isolated_puyo_count = sp2.isolated_puyo_count;
        ctx.p2_near_group_count    = sp2.near_group_count;
        ctx.p2_color_diversity     = sp2.color_diversity;
        ctx.p2_height_variance     = sp2.height_variance;
        ctx.p2_death_col_height    = sp2.death_col_height;
        ctx.p2_buried_puyo_count   = sp2.buried_puyo_count;
        ctx.p2_ojama_dropped       = sp2.ojama_dropped;
        ctx.p2_pending_ojama       = sp2.pending_ojama;
        ctx.p2_potential_chain     = sp2.potential_chain;
        ctx.p2_potential_score     = sp2.potential_score;
    }

    return ctx;
}
} // namespace puyotan
