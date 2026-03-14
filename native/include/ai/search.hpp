#pragma once

#include "engine/game_state.hpp"

namespace puyotan {

// プレイヤーが行う着手（列選択）を表す
struct Move {
    int column;
};

// 与えられたゲーム状態から、AIが最適な着手を選択する
//
// @param state 現在のゲーム状態
// @return 選択された着手
Move chooseMove(const GameState& state);

}
