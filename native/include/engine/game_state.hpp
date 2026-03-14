#pragma once

#include "engine/board.hpp"
#include "config/engine_config.hpp"

namespace puyotan {

// GameState: 盤面全体のスナップショット
// Cell enumはboard.hppで定義済み。重複定義を避けるためここでは定義しない。
struct GameState {
    Board board;
};

} // namespace puyotan
