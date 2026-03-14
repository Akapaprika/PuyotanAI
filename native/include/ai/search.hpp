#pragma once

#include "engine/board.hpp"

namespace puyotan {

// プレイヤーが行う着手（列選択と回転）を表す
struct Move {
    int x;
    Rotation rotation;
};

// 与えられた盤面状態から、AIが最適な着手を選択する
//
// @param board 現在の盤面
Move chooseMove(const Board& board);

}
