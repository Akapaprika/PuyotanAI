#pragma once

#include <puyotan/engine/match.hpp>
#include <cstdint>

namespace puyotan {

class ObservationBuilder {
public:
    static constexpr std::size_t kBytesPerCol = 14; 
    static constexpr std::size_t kBytesPerColor = 6 * 14;
    static constexpr std::size_t kBytesPerField = 5 * 6 * 14;
    static constexpr std::size_t kBytesPerObservation = 2 * kBytesPerField;

    // Determines the fixed color mapping for the match
    static void compute_color_map(const PuyotanMatch& m, int p_idx, uint8_t color_map[5]);
    
    // Renders the bitboards into the [5, 6, 14] observation tensor
    static void render_field(const Board& field, const uint8_t color_map[5], uint8_t* dst_player_obs, bool mask_row12);
    
    // Builds the full observation for both players including Row 13 metadata
    static void build_observation(const PuyotanMatch& m, uint8_t* obs_ptr);
};

} // namespace puyotan
