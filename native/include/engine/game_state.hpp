#pragma once
#include <array>
#include <cstdint>  // for std::uint8_t used in Cell enum
#include "../config/engine_config.hpp"
#include "board.hpp"  // for Cell enum

namespace puyotan {

struct GameState {
    std::array<std::array<Cell, config::Board::kWidth>, config::Board::kHeight> board;
};

}
