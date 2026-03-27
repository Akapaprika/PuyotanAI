#include <puyotan/game/puyotan_vector_match.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace puyotan {

namespace {

// SSE4.1 bit expansion table (0-255 -> 64-bit sparse)
#include "expand_table_data.inc"

/**
 * Maps RL Action Index [0-21] to game Action [Column, Rotation].
 * Layout:
 *   [0-5]   -> Up, Col 0-5
 *   [6-11]  -> Right, Col 0-5
 *   [12-17] -> Down, Col 0-5
 *   [18-21] -> Left, Col 0-3 (Columns 4,5 are restricted in Left rotation to prevent wall-clip errors)
 */
Action GET_ACTION(int idx) {
    if (idx < 0 || idx >= 22) return Action{ActionType::PASS};
    
    int rot_idx = idx / 6;
    int col = idx % 6;
    Rotation rot = static_cast<Rotation>(rot_idx);
    
    // NOTE: Fixed bug: Use ActionType::PUT for actual moves.
    return Action{ActionType::PUT, static_cast<int8_t>(col), rot};
}

/**
 * Normalizes colors based on NEXT pieces (priority) and field presence.
 */
void compute_color_map(const PuyotanMatch& m, int p_idx, uint8_t color_map[5]) {
    memset(color_map, 0, 5);
    uint8_t next_id = 1;
    auto map_one = [&](Cell c) {
        int ci = static_cast<int>(c);
        if (ci >= 4) return;
        if (color_map[ci] == 0) color_map[ci] = next_id++;
    };
    for (int off = 0; off < 3; ++off) {
        PuyoPiece next = m.getPiece(p_idx, off);
        map_one(next.axis);
        map_one(next.sub);
    }
    const auto& board = m.getPlayer(p_idx).field;
    for (int c = 0; c < 4; ++c) {
        if (color_map[c] == 0 && !board.getBitboard(static_cast<Cell>(c)).empty()) {
            color_map[c] = next_id++;
        }
    }
}

void render_field(const Board& field, const uint8_t color_map[5], uint8_t* dst_player_obs, bool mask_row12) {
    static constexpr std::size_t kBytesPerCol = 14; 
    static constexpr std::size_t kBytesPerColor = 6 * 14;
    uint16_t row_mask = mask_row12 ? 0x0FFF : 0x1FFF;

    auto write_col = [&](uint8_t* color_base, int x, uint16_t col_data) {
        uint8_t* dst = color_base + x * kBytesPerCol;
        uint64_t lo = kExpandTable[col_data & 0xFF];
        uint64_t hi = kExpandTable[(col_data >> 8) & 0x3F];
        std::memcpy(dst, &lo, 8);
        std::memcpy(dst + 8, &hi, 6);
    };

    for (int c = 0; c < 4; ++c) {
        uint8_t mapped_idx = color_map[c];
        if (mapped_idx == 0) continue;
        uint8_t* color_base = dst_player_obs + mapped_idx * kBytesPerColor;
        const BitBoard& bb = field.getBitboard(static_cast<Cell>(c));
        for (int x = 0; x < 6; ++x) {
            uint64_t val = (&bb.lo)[x >> 2];
            uint16_t col_data = static_cast<uint16_t>(val >> ((x & 3) << 4)) & row_mask;
            write_col(color_base, x, col_data);
        }
    }
    uint8_t* ojama_base = dst_player_obs + 0 * kBytesPerColor;
    const BitBoard& oj_bb = field.getBitboard(Cell::Ojama);
    for (int x = 0; x < 6; ++x) {
        uint64_t val = (&oj_bb.lo)[x >> 2];
        uint16_t col_data = static_cast<uint16_t>(val >> ((x & 3) << 4)) & row_mask;
        write_col(ojama_base, x, col_data);
    }
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
    float* rew_ptr = static_cast<float*>(rewards.mutable_data());
    bool* term_ptr = static_cast<bool*>(terminated.mutable_data());
    int* chain_ptr = static_cast<int*>(chains.mutable_data());

    {
        pybind11::gil_scoped_release release;
        static constexpr bool kTermTable[] = {true, false, true, true};

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            auto& m = matches_[i];
            
            // 重要: 意思決定フェーズの開始時に前ターンのスコアをリセット
            // これにより、お邪魔が降るだけのターン（PUTが発生しないターン）での報酬重複を防止
            const_cast<PuyotanPlayer&>(m.getPlayer(0)).last_score = 0;
            const_cast<PuyotanPlayer&>(m.getPlayer(1)).last_score = 0;

            m.setAction(0, GET_ACTION(p1_ptr[i]));
            if (p2_ptr) m.setAction(1, GET_ACTION(p2_ptr[i]));
            else m.setAction(1, Action{ActionType::PASS});

            while (m.getStatus() == MatchStatus::PLAYING) {
                int mask = m.stepUntilDecision();
                if (mask == 3 || mask == 0) break;
                if ((mask & 1) != 0) m.setAction(0, Action{ActionType::PASS});
                if ((mask & 2) != 0) m.setAction(1, Action{ActionType::PASS});
            }

            const auto& p1 = m.getPlayer(0);
            // 1手ごとのペナルティ (-0.05) + お邪魔報酬 (120個 = 10pt)
            float r = -0.05f + (float)p1.last_score / 840.0f;

            MatchStatus status = m.getStatus();
            if (status == MatchStatus::WIN_P1) r += 10.0f;
            else if (status == MatchStatus::WIN_P2) r -= 10.0f;
            else if (status == MatchStatus::DRAW) r -= 5.0f;

            rew_ptr[i] = r;
            chain_ptr[i] = p1.last_chain_count;
            bool is_term = kTermTable[static_cast<uint8_t>(status)];
            term_ptr[i] = is_term;

            if (is_term) {
                m = PuyotanMatch(base_seed_ + i);
                m.start(); // Fixed: start() in auto-reset
                m.stepUntilDecision(); // Fixed: stepUntilDecision in auto-reset
            }
        }
    }
    return pybind11::make_tuple(getObservationsAll(out_obs), std::move(rewards), std::move(terminated), std::move(chains));
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
        pybind11::gil_scoped_release release;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            uint8_t* obs_ptr = out_base + i * kBytesPerObservation;
            memset(obs_ptr, 0, kBytesPerObservation);
            const auto& m = matches_[i];
            const auto& p0 = m.getPlayer(0);
            const auto& p1 = m.getPlayer(1);

            uint8_t color_map[5];
            compute_color_map(m, 0, color_map);
            render_field(p0.field, color_map, obs_ptr + 0 * kBytesPerField, false);
            render_field(p1.field, color_map, obs_ptr + 1 * kBytesPerField, true);

            auto write_meta = [&](uint8_t* f_ptr, int x, uint8_t mapped_val, uint8_t value = 1) {
                if (mapped_val == 0) return;
                f_ptr[mapped_val * kBytesPerColor + x * kBytesPerCol + 13] = value;
            };

            for (int off = 0; off < 3; ++off) {
                PuyoPiece next = m.getPiece(0, off);
                write_meta(obs_ptr, off * 2,     color_map[static_cast<int>(next.axis)]);
                write_meta(obs_ptr, off * 2 + 1, color_map[static_cast<int>(next.sub)]);
            }

            uint8_t* self_ojama_meta = obs_ptr + 0 * kBytesPerField + 0 * kBytesPerColor + 13;
            self_ojama_meta[0 * kBytesPerCol] = (uint8_t)std::min((int)p0.active_ojama, 255);
            self_ojama_meta[1 * kBytesPerCol] = (uint8_t)std::min((int)p0.non_active_ojama, 255);
            uint8_t* opp_ojama_meta = obs_ptr + 1 * kBytesPerField + 0 * kBytesPerColor + 13;
            opp_ojama_meta[0 * kBytesPerCol] = (uint8_t)std::min((int)p1.active_ojama, 255);
            opp_ojama_meta[1 * kBytesPerCol] = (uint8_t)std::min((int)p1.non_active_ojama, 255);
        }
    }
    return arr;
}

} // namespace puyotan
