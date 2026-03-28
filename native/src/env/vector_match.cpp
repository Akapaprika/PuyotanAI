#include <puyotan/env/vector_match.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <algorithm>
#include <cstring>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/reward.hpp>
#include <immintrin.h>

namespace puyotan {

namespace {

/**
 * Maps RL Action Index [0-21] to game Action [Column, Rotation].
 * Layout:
 *   [0-5]   -> Up, Col 0-5
 *   [6-11]  -> Right, Col 0-5
 *   [12-17] -> Down, Col 0-5
 *   [18-21] -> Left, Col 0-3 (Columns 4,5 are restricted in Left rotation to prevent wall-clip errors)
 */
Action GET_ACTION(int idx) {
    if (idx < 0 || idx >= 22) return Action{ActionType::Pass};
    if (idx < 6) return Action{ActionType::Put, static_cast<int8_t>(idx), Rotation::Up};
    if (idx < 11) return Action{ActionType::Put, static_cast<int8_t>(idx - 6), Rotation::Right}; // Col 0-4
    if (idx < 17) return Action{ActionType::Put, static_cast<int8_t>(idx - 11), Rotation::Down}; // Col 0-5
    return Action{ActionType::Put, static_cast<int8_t>(idx - 16), Rotation::Left}; // Col 1-5
}
} // anonymous namespace

PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, uint32_t base_seed) 
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
    for (int i = 0; i < num_matches; ++i) {
        matches_.emplace_back(base_seed + i);
        matches_.back().start();
        matches_.back().stepUntilDecision();
    }
}

void PuyotanVectorMatch::reset(int id) noexcept {
    if (id == -1) {
        #pragma omp parallel for
        for (int i = 0; i < (int)matches_.size(); ++i) {
            matches_[i] = PuyotanMatch(base_seed_ + i);
            matches_[i].start(); // Fixed: Must call start()
            matches_[i].stepUntilDecision(); // Fixed: Advance to decision state
        }
    } else {
        matches_[id] = PuyotanMatch(base_seed_ + id);
        matches_[id].start(); // Fixed: Must call start()
        matches_[id].stepUntilDecision(); // Fixed: Advance to decision state
    }
}

std::vector<int> PuyotanVectorMatch::stepUntilDecision() {
    std::vector<int> masks(matches_.size());
    #pragma omp parallel for
    for (int i = 0; i < (int)matches_.size(); ++i) {
        masks[i] = matches_[i].stepUntilDecision();
    }
    return masks;
}

void PuyotanVectorMatch::setActions(const std::vector<int>& match_indices, 
                                     const std::vector<int>& player_ids,
                                     const std::vector<Action>& actions) {
    for (size_t i = 0; i < match_indices.size(); ++i) {
        matches_[match_indices[i]].setAction(player_ids[i], actions[i]);
    }
}

pybind11::tuple PuyotanVectorMatch::step(pybind11::array_t<int> p1_actions, 
                                         std::optional<pybind11::array_t<int>> p2_actions,
                                         std::optional<pybind11::array_t<uint8_t>> out_obs) {
    const int n = static_cast<int>(matches_.size());
    auto p1_ptr = p1_actions.data();
    auto p2_ptr = p2_actions.has_value() ? p2_actions->data() : nullptr;

    pybind11::array_t<float> rewards(n);
    pybind11::array_t<bool> terminated(n);
    pybind11::array_t<int> chains(n);
    pybind11::array_t<int> scores(n);
    float* rew_ptr = static_cast<float*>(rewards.mutable_data());
    bool* term_ptr = static_cast<bool*>(terminated.mutable_data());
    int* chain_ptr = static_cast<int*>(chains.mutable_data());
    int* score_ptr = static_cast<int*>(scores.mutable_data());

    {
        pybind11::gil_scoped_release release;
        static constexpr bool kTermTable[] = {true, false, true, true, true};

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            auto& m = matches_[i];
            
            const auto& p1_pre = m.getPlayer(0);
            const auto& p2_pre = m.getPlayer(1);
            int start_score_p1 = p1_pre.score;
            int start_score_p2 = p2_pre.score;

            m.setAction(0, GET_ACTION(p1_ptr[i]));
            if (p2_ptr) m.setAction(1, GET_ACTION(p2_ptr[i]));
            else m.setAction(1, Action{ActionType::Pass});

            while (m.getStatus() == MatchStatus::Playing) {
                int mask = m.stepUntilDecision();
                if (mask == 3 || mask == 0) break;
                if ((mask & 1) != 0) m.setAction(0, Action{ActionType::Pass});
                if ((mask & 2) != 0) m.setAction(1, Action{ActionType::Pass});
            }

            const auto& p1 = m.getPlayer(0);
            const auto& p2 = m.getPlayer(1);
            MatchStatus status = m.getStatus();

            // Helper to compute connectivity score and related metrics
            auto get_board_metrics = [](const Board& board, int& out_conn, int& out_iso, int& out_div) {
                out_conn = 0;
                out_iso = 0;
                out_div = 0;
                for (int c = 0; c < config::Board::kNumColors - 1; ++c) {
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
                    const BitBoard iso = bb.andNot(neighbors);
                    out_iso += iso.popcount();
                }
            };

            // Helper to compute buried puyo count (colored puyos under any ojama)
            auto get_buried_count = [](const Board& board) -> int {
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
                                        board.getBitboard(Cell::Green) | board.getBitboard(Cell::Yellow) | 
                                        board.getBitboard(Cell::Purple);
                return (buried & s).popcount();
            };

            // Helper to compute max chain achievable by adding exactly one puyo
            auto get_max_potential_chain = [](const Board& board) -> int {
                int max_chain = 0;
                // Try each column
                for (int x = 0; x < 6; ++x) {
                    int h = board.getColumnHeight(x);
                    if (h >= 12) continue; // Skip if too high (Row 12 is ghost, 13 is death)
                    
                    // Try each normal color
                    for (int c = 0; c < 4; ++c) {
                        Board temp = board;
                        temp.dropNewPiece(x, h, static_cast<Cell>(c));
                        
                        // Check for immediate erasure
                        ErasureData ed = Chain::findGroups(temp);
                        if (ed.num_erased > 0) {
                            int chain = 0;
                            while (ed.num_erased > 0) {
                                chain++;
                                Chain::applyErasure(temp, ed);
                                Gravity::apply(temp);
                                ed = Chain::findGroups(temp);
                            }
                            if (chain > max_chain) max_chain = chain;
                        }
                    }
                }
                return max_chain;
            };

            RewardContext ctx;
            ctx.p1_delta_score = p1.score - start_score_p1;
            ctx.p1_chain_count = p1.last_chain_count;
            ctx.p1_puyo_count  = p1.field.getOccupied().popcount();
            get_board_metrics(p1.field, ctx.p1_connectivity_score, ctx.p1_isolated_puyo_count, ctx.p1_color_diversity);
            ctx.p1_death_col_height = p1.field.getColumnHeight(config::Rule::kDeathCol);
            ctx.p1_buried_puyo_count  = get_buried_count(p1.field);
            ctx.p1_ojama_dropped      = p1.total_ojama_dropped - p1_pre.total_ojama_dropped;
            ctx.p1_pending_ojama      = p1.active_ojama + p1.non_active_ojama;
            ctx.p1_potential_chain    = get_max_potential_chain(p1.field);

            ctx.p2_delta_score = p2.score - start_score_p2;
            ctx.p2_chain_count = p2.last_chain_count;
            ctx.p2_puyo_count  = p2.field.getOccupied().popcount();
            get_board_metrics(p2.field, ctx.p2_connectivity_score, ctx.p2_isolated_puyo_count, ctx.p2_color_diversity);
            ctx.p2_death_col_height = p2.field.getColumnHeight(config::Rule::kDeathCol);
            ctx.p2_buried_puyo_count  = get_buried_count(p2.field);
            ctx.p2_ojama_dropped      = p2.total_ojama_dropped - p2_pre.total_ojama_dropped;
            ctx.p2_pending_ojama      = p2.active_ojama + p2.non_active_ojama;
            ctx.p2_potential_chain    = get_max_potential_chain(p2.field);

            ctx.status = status;

            rew_ptr[i] = reward_calc.calculate(ctx, 0);
            chain_ptr[i] = p1.last_chain_count;
            score_ptr[i] = ctx.p1_delta_score;

            bool is_term = kTermTable[static_cast<uint8_t>(status)];
            term_ptr[i] = is_term;

            if (is_term) {
                m = PuyotanMatch(base_seed_ + i);
                m.start();
                m.stepUntilDecision();
            }
        }
    }
    return pybind11::make_tuple(getObservationsAll(out_obs), std::move(rewards), std::move(terminated), std::move(chains), std::move(scores));
}

pybind11::array_t<uint8_t> PuyotanVectorMatch::getObservationsAll(std::optional<pybind11::array_t<uint8_t>> out_obs) const {
    const int n = static_cast<int>(matches_.size());
    static constexpr std::size_t kBytesPerCol = 14; 
    static constexpr std::size_t kBytesPerColor = 6 * 14;
    static constexpr std::size_t kBytesPerField = 5 * 6 * 14;
    static constexpr std::size_t kBytesPerObservation = 2 * kBytesPerField;

    pybind11::array_t<uint8_t> arr;
    if (out_obs.has_value()) arr = *out_obs;
    else arr = pybind11::array_t<uint8_t>({(std::size_t)n, (std::size_t)2, (std::size_t)5, (std::size_t)6, (std::size_t)14});
    uint8_t* out_base = static_cast<uint8_t*>(arr.mutable_data());

    {
        // RELEASE GIL: Batch observation construction is also parallelized via OpenMP.
        pybind11::gil_scoped_release release;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* obs_ptr = out_base + i * ObservationBuilder::kBytesPerObservation;
            ObservationBuilder::buildObservation(matches_[i], obs_ptr);
        }
    }
    return arr;
}

} // namespace puyotan
