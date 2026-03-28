#pragma once

#include <puyotan/engine/match.hpp>
#include <cstdint>

namespace puyotan {

/**
 * @class ObservationBuilder
 * @brief Utilities for converting PuyotanMatch state into PyTorch-ready tensors.
 */
class ObservationBuilder {
public:
    static constexpr std::size_t kBytesPerCol = 14; 
    static constexpr std::size_t kBytesPerColor = 6 * 14;
    static constexpr std::size_t kBytesPerField = 5 * 6 * 14;
    static constexpr std::size_t kBytesPerObservation = 2 * kBytesPerField;

    /**
     * @brief Computes a fixed color mapping for the match.
     * @param m The match state.
     * @param p_idx The player index for whom the map is being built.
     * @param color_map Output array mapping internal color indices to observation indices.
     */
    static void computeColorMap(const PuyotanMatch& m, int p_idx, uint8_t color_map[5]);
    
    /**
     * @brief Renders bitboards into a [5, 6, 14] observation tensor for one player.
     * @param field The player's bitboard field.
     * @param color_map The color index mapping to use.
     * @param dst_player_obs Destination buffer for the tensor.
     * @param mask_row12 Whether to mask out floating pieces in the ghost row.
     */
    static void renderField(const Board& field, const uint8_t color_map[5], uint8_t* dst_player_obs, bool mask_row12);
    
    /**
     * @brief Builds a full observation for both players including Row 13 metadata.
     * @param m The source match.
     * @param obs_ptr Destination buffer of size kBytesPerObservation.
     */
    static void buildObservation(const PuyotanMatch& m, uint8_t* obs_ptr);
};

} // namespace puyotan
