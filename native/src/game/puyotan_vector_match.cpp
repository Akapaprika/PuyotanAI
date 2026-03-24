#include <puyotan/game/puyotan_vector_match.hpp>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstddef>
#include <cstring>
#include <bit>
#include <array>

namespace puyotan {

PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, uint32_t base_seed) 
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
    prev_states_.resize(num_matches);
    for (int i = 0; i < num_matches; ++i) {
        matches_.emplace_back(base_seed + i);
    }
}

void PuyotanVectorMatch::reset(int id) {
    if (id >= 0) {
        matches_[id] = PuyotanMatch(base_seed_ + id);
    } else {
        const int n = static_cast<int>(matches_.size());
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            matches_[i] = PuyotanMatch(base_seed_ + i);
        }
    }
}

std::vector<int> PuyotanVectorMatch::stepUntilDecision() {
    const int n = static_cast<int>(matches_.size());
    std::vector<int> masks(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masks[i] = matches_[i].stepUntilDecision();
    }
    return masks;
}

void PuyotanVectorMatch::setActions(const std::vector<int>& match_indices, 
                                    const std::vector<int>& player_ids,
                                    const std::vector<Action>& actions) {
    const int n = static_cast<int>(match_indices.size());
    // This is typically called from Python and is not heavily parallelizable 
    // unless we have a large number of actions to set.
    for (int i = 0; i < n; ++i) {
        matches_[match_indices[i]].setAction(player_ids[i], actions[i]);
    }
}

namespace {
    struct ActionCode {
        int col;
        Rotation rot;
    };

    constexpr std::array<uint64_t, 256> generateBitExpandTable() {
        std::array<uint64_t, 256> table{};
        for (int i = 0; i < 256; ++i) {
            uint64_t val = 0;
            for (int b = 0; b < 8; ++b) {
                val |= (static_cast<uint64_t>((i >> b) & 1) << (b * 8));
            }
            table[i] = val;
        }
        return table;
    }
    static constexpr auto kExpandTable = generateBitExpandTable();

#include <cassert>
    inline ActionCode decodeAction(int code) {
        assert(code >= 0 && code < 22);
        static constexpr ActionCode kTable[22] = {
            {0, Rotation::Up}, {1, Rotation::Up}, {2, Rotation::Up}, {3, Rotation::Up}, {4, Rotation::Up}, {5, Rotation::Up},
            {0, Rotation::Right}, {1, Rotation::Right}, {2, Rotation::Right}, {3, Rotation::Right}, {4, Rotation::Right},
            {0, Rotation::Down}, {1, Rotation::Down}, {2, Rotation::Down}, {3, Rotation::Down}, {4, Rotation::Down}, {5, Rotation::Down},
            {1, Rotation::Left}, {2, Rotation::Left}, {3, Rotation::Left}, {4, Rotation::Left}, {5, Rotation::Left}
        };
        return kTable[code];
    }
}

pybind11::tuple PuyotanVectorMatch::step(pybind11::array_t<int> p1_actions, std::optional<pybind11::array_t<int>> p2_actions) {
    auto req1 = p1_actions.request();
    int* p1_ptr = static_cast<int*>(req1.ptr);
    
    // Branchless optional p2: if nullopt, point to a dummy -1 with 0-stride.
    static const int kNoAction = -1;
    int* p2_ptr = const_cast<int*>(&kNoAction);
    int p2_stride = 0;

    if (p2_actions.has_value()) {
        p2_ptr = static_cast<int*>(p2_actions->request().ptr);
        p2_stride = 1;
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
            ActionCode c1 = decodeAction(act1);
            m.setAction(0, Action(ActionType::PUT, c1.col, c1.rot));

            // Branchless retrieval: i*1 if present, i*0 (always kNoAction) if not.
            int act2 = p2_ptr[i * p2_stride];
            
            // Extract sign bit: 0 if act2 >= 0 (PUT), 1 if act2 < 0 (PASS)
            ActionType t2 = static_cast<ActionType>(static_cast<unsigned int>(act2) >> 31);
            
            // Compute max(act2, 0) arithmetically to safely index LUT
            int safe_act2 = act2 & ~(act2 >> 31);
            
            ActionCode c2 = decodeAction(safe_act2);
            m.setAction(1, Action(t2, c2.col, c2.rot));

            m.stepNextFrame();
            m.stepUntilDecision();

            const auto& p1 = m.getPlayer(0);
            float r = 0.0f;
            r += (p1.score - prev_states_[i].score) * 0.002f;
            r -= (p1.active_ojama - prev_states_[i].ojama) * 0.001f;

            // Branchless match result reward and termination check
            static constexpr float kMatchRewards[] = { 0.0f, 0.0f, 10.0f, -10.0f, 0.0f };
            static constexpr bool kTermTable[]     = { false, false, true, true, true };
            
            const MatchStatus status = m.getStatus();
            r += kMatchRewards[static_cast<uint8_t>(status)];

            rew_ptr[i] = r;
            chain_ptr[i] = static_cast<int>(p1.chain_count);

            bool is_term = kTermTable[static_cast<uint8_t>(status)];
            term_ptr[i] = is_term;

            prev_states_[i].score = p1.score;
            prev_states_[i].ojama = p1.active_ojama;

            if (is_term) {
                m = PuyotanMatch(base_seed_ + i);
                m.start();
                m.stepUntilDecision();
                prev_states_[i].score = m.getPlayer(0).score;
                prev_states_[i].ojama = m.getPlayer(0).active_ojama;
            }
        }
    }

    return pybind11::make_tuple(getObservationsAll(), std::move(rewards), std::move(terminated), std::move(chains));
}

#include <immintrin.h>

pybind11::array_t<uint8_t> PuyotanVectorMatch::getObservationsAll() const {
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

        const std::size_t total_bytes = (std::size_t)n * kBytesPerMatch;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* match_ptr = out_base + i * kBytesPerMatch;
            
            // Zero initialize this thread's specific memory chunk (NUMA First-Touch & Parallelization)
            memset(match_ptr, 0, kBytesPerMatch);

            for (int p_idx = 0; p_idx < config::Rule::kNumPlayers; ++p_idx) {
                const auto& player = matches_[i].getPlayer(p_idx);
                uint8_t* player_ptr = match_ptr + p_idx * kBytesPerPlayer;
                for (int c = 0; c < colors; ++c) {
                    uint8_t* color_ptr = player_ptr + c * kBytesPerColor;
                    BitBoard bb = player.field.getBitboard(static_cast<Cell>(c));

                    bb.lo &= config::Board::kLoMask;
                    bb.hi &= config::Board::kHiMask;
                    
                    // SWAR (SIMD Within A Register) array layout synthesis
                    // Completely eliminates unpredictable branches of while(bb.lo) bit scans
                    auto write_col = [&](int x, uint16_t col_data) {
                        uint8_t* dst = color_ptr + x * kBytesPerCol;
                        uint64_t lo = kExpandTable[col_data & 0xFF];
                        uint64_t hi = kExpandTable[(col_data >> 8) & 0x1F]; // Read up to 5 bits (13-8=5)
                        
                        // 13-bytes contiguous unaligned writes
                        std::memcpy(dst, &lo, 8);
                        std::memcpy(dst + 8, &hi, 5);
                    };

                    // Extract all 6 layout columns systematically
                    write_col(0, static_cast<uint16_t>(bb.lo & 0xFFFF));
                    write_col(1, static_cast<uint16_t>((bb.lo >> 16) & 0xFFFF));
                    write_col(2, static_cast<uint16_t>((bb.lo >> 32) & 0xFFFF));
                    write_col(3, static_cast<uint16_t>((bb.lo >> 48) & 0xFFFF));
                    write_col(4, static_cast<uint16_t>(bb.hi & 0xFFFF));
                    write_col(5, static_cast<uint16_t>((bb.hi >> 16) & 0xFFFF));
                }
            }
        }
    }
    
    return arr;
}

} // namespace puyotan
