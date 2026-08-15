#pragma once

#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/scorer.hpp>

namespace puyotan::search {

/**
 * @brief Computes the maximum potential chain score achievable by placing
 *        one additional colored puyo adjacent to an existing group on the board.
 *
 * Shared by SoloBeamEvaluator and VsBeamEvaluator to eliminate duplicated code.
 * Marked [[nodiscard]] and noexcept to match evaluator contracts.
 *
 * @param board   The board state to evaluate.
 * @param heights Precomputed per-column heights (avoids recomputing inside loops).
 *                Must be an array of exactly config::Board::kWidth elements.
 * @return Maximum single-step chain score achievable by one probe placement.
 *         Returns 0 if no chain can be triggered.
 */
[[nodiscard]] inline int computeMaxPotentialScore(
    const Board& board,
    const int    heights[config::Board::kWidth]) noexcept
{
    int max_pot_score = 0;

    for (int x = 0; x < config::Board::kWidth; ++x) {
        const int h = heights[x];
        if (h >= config::Board::kChainableRows)
            continue;

        for (int c = 0; c < config::Rule::kColors; ++c) {
            const BitBoard& bb = board.getBitboard(static_cast<Cell>(c));

            // SIMD neighbor mask: 4 shifts replace 3 conditional bb.get() calls.
            const BitBoard neighbor =
                bb.shiftUpRaw() | bb.shiftDownRaw() |
                bb.shiftLeftRaw() | bb.shiftRightRaw();
            if (!neighbor.get(x, h))
                continue;

            Board temp = board;
            temp.dropNewPiece(x, h, static_cast<Cell>(c));

            ErasureData ed;
            Chain::scanGroups(temp, ed, 1u << c);
            if (ed.num_erased == 0)
                continue;

            int pot_chain = 0, pot_score = 0;
            while (ed.num_erased > 0) {
                ++pot_chain;
                pot_score += Scorer::calculateStepScore(ed, pot_chain);
                Chain::applyErasure(temp, ed);
                uint32_t fallen = Gravity::execute(temp);
                Chain::scanGroups(temp, ed, fallen);
            }
            if (pot_score > max_pot_score)
                max_pot_score = pot_score;
        }
    }

    return max_pot_score;
}

} // namespace puyotan::search
