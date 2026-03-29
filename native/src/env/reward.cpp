#include <puyotan/env/reward.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <immintrin.h>

namespace puyotan {

namespace {

// Helper to compute connectivity score and related metrics
void get_board_metrics(const Board& board, int& out_conn, int& out_iso, int& out_div) {
    out_conn = 0;
    out_iso = 0;
    out_div = 0;
    for (int c = 0; c < config::Rule::kColors; ++c) {
        const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));
        if (bb.empty()) continue;
        
        const BitBoard U = bb.shiftUpRaw();
        const BitBoard D = bb.shiftDownRaw();
        const BitBoard L = bb.shiftLeftRaw();
        const BitBoard R = bb.shiftRightRaw();
        const BitBoard neighbors = U | D | L | R;

        // Connectivity: puyos with >= 2 neighbors of same color
        const BitBoard has_2 = bb & ((U & D) | (L & R) | ((U | D) & (L | R)));
        int conn = has_2.popcount();
        out_conn += conn;
        if (conn > 0) out_div++;

        // Isolation: puyos with 0 neighbors of same color
        BitBoard iso = bb;
        iso.andNot(neighbors);
        out_iso += iso.popcount();
    }
}

// Helper to compute buried puyo count (colored puyos under any ojama)
int get_buried_count(const Board& board) {
    const BitBoard& oj = board.getBitboard(Cell::Ojama);
    if (oj.empty()) return 0;
    
    // Smear ojama downwards using 4 shifts (2^0, 2^1, 2^2, 2^3 = 15 rows)
    BitBoard s = oj;
    s |= s.shiftDownRaw();
    s |= _mm_srli_epi64(s.m128, 2);
    s |= _mm_srli_epi64(s.m128, 4);
    s |= _mm_srli_epi64(s.m128, 8);

    // Buried = (Any non-ojama puyo) AND (Smeared ojama mask)
    const BitBoard buried = board.getBitboard(Cell::Red) | board.getBitboard(Cell::Blue) | 
                            board.getBitboard(Cell::Green) | board.getBitboard(Cell::Yellow);
    return (buried & s).popcount();
}

// Helper to compute max chain achievable by adding exactly one puyo
int get_max_potential_chain(const Board& board) {
    int max_chain = 0;
    // Try each column
    for (int x = 0; x < config::Board::kWidth; ++x) {
        int h = board.getColumnHeight(x);
        if (h >= config::Board::kChainableRows) continue; // Skip if too high (Row 12 is ghost, 13 is death)
        
        // Try each normal color
        for (int c = 0; c < config::Rule::kColors; ++c) {
            Board temp = board;
            temp.dropNewPiece(x, h, static_cast<Cell>(c));
            
            // OPTIMIZATION: Check for immediate erasure ONLY for the color we just dropped
            ErasureData ed = Chain::findGroups(temp, 1u << c);
            if (ed.num_erased > 0) {
                int chain = 0;
                while (ed.num_erased > 0) {
                    chain++;
                    Chain::applyErasure(temp, ed);
                    // OPTIMIZATION: execute() returns the mask of colors that actually fell.
                    // We ONLY need to search these moved colors for the next chain link!
                    uint32_t fallen_mask = Gravity::execute(temp);
                    ed = Chain::findGroups(temp, fallen_mask);
                }
                if (chain > max_chain) max_chain = chain;
            }
        }
    }
    return max_chain;
}

} // namespace

RewardContext RewardCalculator::extractContext(const PuyotanMatch& m, 
                                             int start_score_p1, int start_score_p2,
                                             int pre_ojama_p1, int pre_ojama_p2) const {
    RewardContext ctx;
    ctx.status = m.getStatus();

    const auto& p1 = m.getPlayer(0);
    ctx.p1_delta_score = p1.score - start_score_p1;
    ctx.p1_chain_count = p1.last_chain_count;
    ctx.p1_puyo_count  = p1.field.getOccupied().popcount();
    get_board_metrics(p1.field, ctx.p1_connectivity_score, ctx.p1_isolated_puyo_count, ctx.p1_color_diversity);
    ctx.p1_death_col_height = p1.field.getColumnHeight(config::Rule::kDeathCol);
    ctx.p1_buried_puyo_count  = get_buried_count(p1.field);
    ctx.p1_ojama_dropped      = p1.total_ojama_dropped - pre_ojama_p1;
    ctx.p1_pending_ojama      = p1.active_ojama + p1.non_active_ojama;
    ctx.p1_potential_chain    = get_max_potential_chain(p1.field);

    const auto& p2 = m.getPlayer(1);
    ctx.p2_delta_score = p2.score - start_score_p2;
    ctx.p2_chain_count = p2.last_chain_count;
    ctx.p2_puyo_count  = p2.field.getOccupied().popcount();
    get_board_metrics(p2.field, ctx.p2_connectivity_score, ctx.p2_isolated_puyo_count, ctx.p2_color_diversity);
    ctx.p2_death_col_height = p2.field.getColumnHeight(config::Rule::kDeathCol);
    ctx.p2_buried_puyo_count  = get_buried_count(p2.field);
    ctx.p2_ojama_dropped      = p2.total_ojama_dropped - pre_ojama_p2;
    ctx.p2_pending_ojama      = p2.active_ojama + p2.non_active_ojama;
    ctx.p2_potential_chain    = get_max_potential_chain(p2.field);

    return ctx;
}

} // namespace puyotan
