#include <pybind11/pybind11.h>
#include "../include/ai/search.hpp"
#include "../include/engine/game_state.hpp"

using namespace puyotan;

PYBIND11_MODULE(puyotan_native, m) {
    pybind11::class_<GameState>(m, "GameState").def(pybind11::init<>());

    pybind11::class_<Move>(m, "Move").def_readwrite("column", &Move::column);

    m.def("choose_move", &choose_move);
}
