#pragma once
#include <array>
#include <cstdint>  // for std::uint8_t used in Cell enum
#include "../config/engine_config.hpp"

namespace puyotan {

enum class Cell : std::uint8_t {
    Empty = 0,
    Red,
    Green,
    Blue,
    Yellow
};

struct GameState {
    std::array<std::array<Cell, config::Board::kWidth>, config::Board::kHeight> board;
};

}
