#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/tsumo.hpp>
#include <puyotan/game/simulator.hpp>
#include <puyotan/game/scorer.hpp>

namespace puyotan {

PYBIND11_MODULE(puyotan_native, m) {
    m.doc() = "Puyotan AI Native Engine (Architectural Refactor)";

    /**
     * @brief Bind common types
     */
    pybind11::enum_<Cell>(m, "Cell")
        .value("Red",    Cell::Red)
        .value("Green",  Cell::Green)
        .value("Blue",   Cell::Blue)
        .value("Yellow", Cell::Yellow)
        .value("Ojama",  Cell::Ojama)
        .value("Empty",  Cell::Empty)
        .export_values();

    pybind11::enum_<Rotation>(m, "Rotation")
        .value("Up",    Rotation::Up)
        .value("Right", Rotation::Right)
        .value("Down",  Rotation::Down)
        .value("Left",  Rotation::Left)
        .export_values();

    pybind11::class_<PuyoPiece>(m, "PuyoPiece")
        .def(pybind11::init<>())
        .def(pybind11::init<Cell, Cell>())
        .def_readwrite("axis", &PuyoPiece::axis)
        .def_readwrite("sub",  &PuyoPiece::sub);

    /**
     * @brief Bind Core logic
     */
    pybind11::class_<BitBoard>(m, "BitBoard")
        .def(pybind11::init<>())
        .def(pybind11::init<uint64_t, uint64_t>())
        .def_readwrite("lo", &BitBoard::lo)
        .def_readwrite("hi", &BitBoard::hi)
        .def("get",   &BitBoard::get)
        .def("set",   &BitBoard::set)
        .def("clear", &BitBoard::clear)
        .def("empty", &BitBoard::empty);

    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get", &Board::get)
        .def("set", &Board::set)
        .def("clear", &Board::clear)
        .def("placePiece", &Board::placePiece)
        .def("getBitboard", &Board::getBitboard)
        .def("getOccupied", &Board::getOccupied);

    pybind11::class_<ErasureData>(m, "ErasureData")
        .def_readwrite("erased",      &ErasureData::erased)
        .def_readwrite("num_erased",  &ErasureData::num_erased)
        .def_readwrite("num_colors",  &ErasureData::num_colors)
        .def_readwrite("num_groups",  &ErasureData::num_groups)
        .def_property_readonly("group_sizes", &ErasureData::group_sizes_vec);

    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", &Chain::execute,
            pybind11::arg("board"), pybind11::arg("color_mask") = 0x0F)
        .def_static("findGroups", &Chain::findGroups);

    /**
     * @brief Bind Game logic
     */
    pybind11::class_<Scorer>(m, "Scorer")
        .def_static("calculateStepScore", &Scorer::calculateStepScore);

    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("get", &Tsumo::get)
        .def("setSeed", &Tsumo::setSeed)
        .def_property_readonly("seed", &Tsumo::getSeed);

    pybind11::class_<Simulator>(m, "Simulator")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("step", &Simulator::step)
        .def("reset", &Simulator::reset)
        .def("getCurrentPiece", &Simulator::getCurrentPiece)
        .def("runBatch", &Simulator::runBatch,
             pybind11::arg("num_games"), pybind11::arg("seed") = 1,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def_property_readonly("is_game_over", &Simulator::isGameOver)
        .def_property_readonly("board", &Simulator::getBoard)
        .def_property_readonly("tsumo", &Simulator::getTsumo)
        .def_property_readonly("tsumo_index", &Simulator::getTsumoIndex)
        .def_property_readonly("total_score", &Simulator::getTotalScore);
}

} // namespace puyotan
