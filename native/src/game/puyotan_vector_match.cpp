#include <puyotan/game/puyotan_vector_match.hpp>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstddef>
#include <cstring>
#include <bit>

namespace puyotan {

PuyotanVectorMatch::PuyotanVectorMatch(int num_matches, int32_t base_seed) 
    : base_seed_(base_seed) {
    matches_.reserve(num_matches);
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

#include <immintrin.h>

pybind11::array_t<uint8_t> PuyotanVectorMatch::get_observations_all() const {
    const int n = static_cast<int>(matches_.size());
    const int colors = config::Board::kNumColors;
    const int width  = config::Board::kWidth;
    const int height = config::Board::kHeight;

    // 1. Allocate array (Must hold GIL, default uninitialized or zeros depending on platform)
    pybind11::array_t<uint8_t> arr({(std::size_t)n, (std::size_t)config::Rule::kNumPlayers, (std::size_t)colors, (std::size_t)width, (std::size_t)height});

    // 2. Parallel processing (Release GIL)
    {
        pybind11::gil_scoped_release release;
        uint8_t* out_base = static_cast<uint8_t*>(arr.mutable_data());
        
        static constexpr std::size_t kBytesPerCol = config::Board::kHeight;
        static constexpr std::size_t kBytesPerColor = config::Board::kWidth * kBytesPerCol;
        static constexpr std::size_t kBytesPerPlayer = config::Board::kNumColors * kBytesPerColor;
        static constexpr std::size_t kBytesPerMatch = config::Rule::kNumPlayers * kBytesPerPlayer;

        // Zero all memory first (highly optimized memset is faster than multidimensional looping)
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

                    // Pre-masking ensures we never extract bits beyond the legal height.
                    bb.lo &= config::Board::kLoMask;
                    bb.hi &= config::Board::kHiMask;
                    
                    // Fast bit extraction! Skip empty spaces entirely.
                    while (bb.lo) {
                        int bit = static_cast<int>(_tzcnt_u64(bb.lo));
                        int x = bit >> 4; // x = bit / 16
                        int y = bit & 0x0F;
                        assert(y < height && "Puyo found outside of legal observation height");
                        color_ptr[x * kBytesPerCol + y] = 1;
                        bb.lo &= bb.lo - 1; // clear lowest set bit
                    }
                    while (bb.hi) {
                        int bit = static_cast<int>(_tzcnt_u64(bb.hi));
                        int x = (bit >> 4) + config::Board::kColsInLo;
                        int y = bit & 0x0F;
                        assert(y < height && "Puyo found outside of legal observation height");
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
