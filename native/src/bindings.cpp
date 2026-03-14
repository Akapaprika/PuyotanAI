#include <pybind11/stl.h>
#include "../include/engine/board.hpp"
#include "../include/engine/gravity.hpp"
#include "../include/engine/chain.hpp"
#include "../include/engine/tsumo.hpp"
#include "../include/engine/simulator.hpp"

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

    // ---- Rotation enum ----
    pybind11::enum_<Rotation>(m, "Rotation")
        .value("Up",    Rotation::Up)
        .value("Right", Rotation::Right)
        .value("Down",  Rotation::Down)
        .value("Left",  Rotation::Left)
        .export_values();

    // ---- PuyoPiece (Struct) ----
    pybind11::class_<PuyoPiece>(m, "PuyoPiece")
        .def(pybind11::init<>())
        .def_readwrite("axis", &PuyoPiece::axis)
        .def_readwrite("sub",  &PuyoPiece::sub);

    // ---- Tsumo ----
    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("get",      &Tsumo::get)
        .def("setSeed",  &Tsumo::setSeed)
        .def("getSeed",  &Tsumo::getSeed);

    // ---- Simulator ----
    pybind11::class_<Simulator>(m, "Simulator")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("step",             &Simulator::step, pybind11::arg("move_x"), pybind11::arg("move_rotation"))
        .def("reset",            &Simulator::reset)
        .def("isGameOver",       &Simulator::isGameOver)
        .def("getTotalScore",    &Simulator::getTotalScore)
        .def("getBoard",         &Simulator::getBoard, pybind11::return_value_policy::reference_internal)
        .def("getTsumoIndex",    &Simulator::getTsumoIndex)
        .def("getCurrentPiece",  &Simulator::getCurrentPiece);

    // ---- Board ----
    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get",         &Board::get)
        .def("set",         &Board::set)
        .def("clear",       &Board::clear);

    // ---- Gravity ----
    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    // ---- Chain ----
    pybind11::class_<Chain::StepResult>(m, "StepResult")
        .def_readwrite("erased",     &Chain::StepResult::erased)
        .def_readwrite("num_erased", &Chain::StepResult::num_erased)
        .def_readwrite("num_colors", &Chain::StepResult::num_colors)
        .def_readwrite("score",      &Chain::StepResult::score);

    pybind11::class_<Chain::ChainResult>(m, "ChainResult")
        .def_readwrite("total_score",  &Chain::ChainResult::total_score)
        .def_readwrite("max_chain",    &Chain::ChainResult::max_chain)
        .def_readwrite("total_erased", &Chain::ChainResult::total_erased);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", [](Board& b, int cn, uint8_t mask) { return Chain::execute(b, cn, mask); },
            pybind11::arg("board"), pybind11::arg("chain_number"), pybind11::arg("color_mask") = 0x0F)
        .def_static("executeChain", [](Board& b, uint8_t mask) { return Chain::executeChain(b, mask); },
            pybind11::arg("board"), pybind11::arg("first_color_mask") = 0x0F)
        .def_static("findGroups",   &Chain::findGroups);
}
