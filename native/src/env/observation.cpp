#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <puyotan/env/observation.hpp>

namespace puyotan {
namespace {
// SSE4.1 bit expansion table (0-255 -> 64-bit sparse)
#include "expand_table_data.inc"
} // namespace

void ObservationBuilder::computeColorMap(const PuyotanMatch& m, int p_idx, uint8_t color_map[5]) {
    memset(color_map, 0, 5);
    uint8_t next_id = 1;
    const Tsumo& tsumo = m.getTsumo();
    // Scan from turn 0 to find the order of appearance for the whole match
    for (int i = 0; i < 256; ++i) {
        PuyoPiece p = const_cast<Tsumo&>(tsumo).get(i);
        auto map_one = [&](Cell c) {
            int ci = static_cast<int>(c);
            if (ci < 0 || ci >= 4)
                return; // Only Normal Colors (0-3)
            if (color_map[ci] == 0)
                color_map[ci] = next_id++;
        };
        map_one(p.axis);
        if (next_id > 4)
            break;
        map_one(p.sub);
        if (next_id > 4)
            break;
    }
}

void ObservationBuilder::renderField(const Board& field, const uint8_t color_map[5], uint8_t* dst_player_obs, bool mask_row12) {
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
        if (mapped_idx == 0)
            continue;
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

void ObservationBuilder::buildObservation(const PuyotanMatch& m, uint8_t* obs_ptr) {
    memset(obs_ptr, 0, kBytesPerObservation);
    const auto& p0 = m.getPlayer(0);
    const auto& p1 = m.getPlayer(1);

    uint8_t color_map[5];
    computeColorMap(m, 0, color_map);
    renderField(p0.field, color_map, obs_ptr + 0 * kBytesPerField, false);
    renderField(p1.field, color_map, obs_ptr + 1 * kBytesPerField, true);

    auto write_meta = [&](uint8_t* f_ptr, int x, uint8_t mapped_val, uint8_t value = 1) {
        if (mapped_val == 0)
            return;
        f_ptr[mapped_val * kBytesPerColor + x * kBytesPerCol + 13] = value;
    };

    for (int off = 0; off < 3; ++off) {
        PuyoPiece next = m.getPiece(0, off);
        write_meta(obs_ptr, off * 2, color_map[static_cast<int>(next.axis)]);
        write_meta(obs_ptr, off * 2 + 1, color_map[static_cast<int>(next.sub)]);
    }

    uint8_t* self_ojama_meta = obs_ptr + 0 * kBytesPerField + 0 * kBytesPerColor + 13;
    self_ojama_meta[0 * kBytesPerCol] = (uint8_t)std::min((int)p0.active_ojama, 255);
    self_ojama_meta[1 * kBytesPerCol] = (uint8_t)std::min((int)p0.non_active_ojama, 255);
    uint8_t* opp_ojama_meta = obs_ptr + 1 * kBytesPerField + 0 * kBytesPerColor + 13;
    opp_ojama_meta[0 * kBytesPerCol] = (uint8_t)std::min((int)p1.active_ojama, 255);
    opp_ojama_meta[1 * kBytesPerCol] = (uint8_t)std::min((int)p1.non_active_ojama, 255);
}
} // namespace puyotan
