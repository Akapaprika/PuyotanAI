#include <pybind11/pybind11.h>
#include "../include/ai/search.hpp"
#include "../include/engine/game_state.hpp"
#include "../include/engine/board.hpp"
#include "../include/engine/gravity.hpp"

using namespace puyotan;

PYBIND11_MODULE(puyotan_native, m) {
    // ---- Cell enum ----
    pybind11::enum_<Cell>(m, "Cell")
        .value("Red",    Cell::Red)
        .value("Green",  Cell::Green)
        .value("Blue",   Cell::Blue)
        .value("Yellow", Cell::Yellow)
        .value("Ojama",  Cell::Ojama)
        .value("Empty",  Cell::Empty)
        .export_values();

    // ---- Board ----
    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get",         &Board::get)
        .def("set",         &Board::set)
        .def("clear",       &Board::clear)
        .def("placePiece",  &Board::placePiece);

    // ---- Gravity ----
    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    // ---- AI interfaces ----
    pybind11::class_<GameState>(m, "GameState").def(pybind11::init<>());
    pybind11::class_<Move>(m, "Move").def_readwrite("column", &Move::column);
    m.def("choose_move", &choose_move);
}
