#pragma once
#include "../engine/game_state.hpp"

namespace puyotan {

struct Move {
    int column;
};

Move choose_move(const GameState& state);

}
