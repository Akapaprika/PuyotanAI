#include "ai/search.hpp"

namespace puyotan {

Move chooseMove(const Board& board) {
    // とりあえず仮実装（3列目、上向きに固定）
    return Move{3, Rotation::Up};
}

}