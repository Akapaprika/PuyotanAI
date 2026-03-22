#include <puyotan/game/puyotan_vector_match.hpp>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstddef>
#include <cstring>
#include <bit>

namespace puyotan {

PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, uint32_t base_seed) 
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
    prev_scores_.assign(num_matches, 0);
    prev_ojama_.assign(num_matches, 0);
    for (int i = 0; i < num_matches; ++i) {
        matches_.emplace_back(std::make_unique<PuyotanMatch>(base_seed + i));
    }
}

void PuyotanVectorMatch::reset(int id) {
    if (id >= 0) {
        *matches_[id] = PuyotanMatch(base_seed_ + id);
    } else {
        const int n = static_cast<int>(matches_.size());
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            *matches_[i] = PuyotanMatch(base_seed_ + i);
        }
    }
}

std::vector<int> PuyotanVectorMatch::step_until_decision() {
    const int n = static_cast<int>(matches_.size());
    std::vector<int> masks(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masks[i] = matches_[i]->stepUntilDecision();
    }
    return masks;
}

void PuyotanVectorMatch::set_actions(const std::vector<int>& match_indices, 
                                    const std::vector<int>& player_ids,
                                    const std::vector<Action>& actions) {
    const int n = static_cast<int>(match_indices.size());
    // This is typically called from Python and is not heavily parallelizable 
    // unless we have a large number of actions to set.
    for (int i = 0; i < n; ++i) {
        matches_[match_indices[i]]->setAction(player_ids[i], actions[i]);
    }
}

namespace {
    struct ActionCode {
        int col;
        Rotation rot;
    };
    inline ActionCode decode_action(int code) {
        if (code < 0) return { 2, Rotation::Up }; 
        if (code < 6) return { code, Rotation::Up };
        if (code < 11) return { code - 6, Rotation::Right };
        if (code < 17) return { code - 11, Rotation::Down };
        if (code < 22) return { code - 16, Rotation::Left };
        return { 2, Rotation::Up };
    }
}

pybind11::tuple PuyotanVectorMatch::step(pybind11::array_t<int> p1_actions, std::optional<pybind11::array_t<int>> p2_actions) {
    auto req1 = p1_actions.request();
    int* p1_ptr = static_cast<int*>(req1.ptr);
    int* p2_ptr = nullptr;

    if (p2_actions.has_value()) {
        p2_ptr = static_cast<int*>(p2_actions->request().ptr);
    }

    const int n = static_cast<int>(matches_.size());
    pybind11::array_t<float> rewards(n);
    pybind11::array_t<bool> terminated(n);
    pybind11::array_t<int> chains(n);

    float* rew_ptr = static_cast<float*>(rewards.mutable_data());
    bool* term_ptr = static_cast<bool*>(terminated.mutable_data());
    int* chain_ptr = static_cast<int*>(chains.mutable_data());

    {
        pybind11::gil_scoped_release release;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            auto& m = matches_[i];

            int act1 = p1_ptr[i];
            ActionCode c1 = decode_action(act1);
            m->setAction(0, Action(ActionType::PUT, c1.col, c1.rot));

            if (p2_ptr) {
                int act2 = p2_ptr[i];
                if (act2 >= 0) {
                    ActionCode c2 = decode_action(act2);
                    m->setAction(1, Action(ActionType::PUT, c2.col, c2.rot));
                } else {
                    m->setAction(1, Action(ActionType::PASS, 0, Rotation::Up));
                }
            } else {
                m->setAction(1, Action(ActionType::PASS, 0, Rotation::Up));
            }

            m->stepNextFrame();
            m->stepUntilDecision();

            const auto& p1 = m->getPlayer(0);
            float r = 0.0f;
            r += (p1.score - prev_scores_[i]) * 0.002f;
            r -= (p1.active_ojama - prev_ojama_[i]) * 0.001f;

            if (m->getStatus() == MatchStatus::WIN_P1) r += 10.0f;
            else if (m->getStatus() == MatchStatus::WIN_P2) r -= 10.0f;

            rew_ptr[i] = r;
            chain_ptr[i] = static_cast<int>(p1.chain_count);

            bool is_term = (m->getStatus() != MatchStatus::PLAYING);
            term_ptr[i] = is_term;

            prev_scores_[i] = p1.score;
            prev_ojama_[i] = p1.active_ojama;

            if (is_term) {
                *m = PuyotanMatch(base_seed_ + i);
                m->start();
                m->stepUntilDecision();
                prev_scores_[i] = m->getPlayer(0).score;
                prev_ojama_[i] = m->getPlayer(0).active_ojama;
            }
        }
    }

    return pybind11::make_tuple(get_observations_all(), std::move(rewards), std::move(terminated), std::move(chains));
}

#include <immintrin.h>

pybind11::array_t<uint8_t> PuyotanVectorMatch::get_observations_all() const {
    const int n = static_cast<int>(matches_.size());
    const int colors = config::Board::kNumColors;
    const int width  = config::Board::kWidth;
    const int height = config::Board::kHeight;

    // 1. Allocate array (Must hold GIL)
    pybind11::array_t<uint8_t> arr({(std::size_t)n, (std::size_t)config::Rule::kNumPlayers, (std::size_t)colors, (std::size_t)width, (std::size_t)height});
    uint8_t* out_base = static_cast<uint8_t*>(arr.mutable_data());

    // 2. Parallel processing (Release GIL)
    {
        pybind11::gil_scoped_release release;
        
        static constexpr std::size_t kBytesPerCol = config::Board::kHeight;
        static constexpr std::size_t kBytesPerColor = config::Board::kWidth * kBytesPerCol;
        static constexpr std::size_t kBytesPerPlayer = config::Board::kNumColors * kBytesPerColor;
        static constexpr std::size_t kBytesPerMatch = config::Rule::kNumPlayers * kBytesPerPlayer;

        // Zero all memory first
        const std::size_t total_bytes = (std::size_t)n * kBytesPerMatch;
        memset(out_base, 0, total_bytes);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* match_ptr = out_base + i * kBytesPerMatch;
            for (int p_idx = 0; p_idx < config::Rule::kNumPlayers; ++p_idx) {
                const auto& player = matches_[i]->getPlayer(p_idx);
                uint8_t* player_ptr = match_ptr + p_idx * kBytesPerPlayer;
                for (int c = 0; c < colors; ++c) {
                    uint8_t* color_ptr = player_ptr + c * kBytesPerColor;
                    BitBoard bb = player.field.getBitboard(static_cast<Cell>(c));

                    bb.lo &= config::Board::kLoMask;
                    bb.hi &= config::Board::kHiMask;
                    
                    // Use a portable bit extraction loop
                    while (bb.lo) {
                        uint64_t lowbit = bb.lo & -bb.lo;
                        // Traditional trailing zero count if BMI1 is not guaranteed
                        // For simplicity and safety on 3020e:
                        int bit = 0;
                        uint64_t v = bb.lo;
                        // This loop is safe and generally fast for sparse bitboards
                        #ifdef _MSC_VER
                            unsigned long index;
                            _BitScanForward64(&index, v);
                            bit = (int)index;
                        #else
                            bit = __builtin_ctzll(v);
                        #endif
                        
                        int x = bit >> 4;
                        int y = bit & 0x0f;
                        color_ptr[x * kBytesPerCol + y] = 1;
                        bb.lo &= bb.lo - 1; 
                    }
                    while (bb.hi) {
                        uint64_t v = bb.hi;
                        int bit = 0;
                        #ifdef _MSC_VER
                            unsigned long index;
                            _BitScanForward64(&index, v);
                            bit = (int)index;
                        #else
                            bit = __builtin_ctzll(v);
                        #endif
                        int x = (bit >> 4) + config::Board::kColsInLo;
                        int y = bit & 0x0f;
                        color_ptr[x * kBytesPerCol + y] = 1;
                        bb.hi &= bb.hi - 1;
                    }
                }
            }
        }
    }
    
    return arr;
}

} // namespace puyotan
