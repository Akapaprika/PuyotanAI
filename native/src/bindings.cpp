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
#include <puyotan/game/puyotan_match.hpp>

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
        .def_property_readonly("erased",     [](const ErasureData& d) { return d.num_erased > 0; })
        .def_readwrite("num_erased",  &ErasureData::num_erased)
        .def_readwrite("num_colors",  &ErasureData::num_colors)
        .def_readwrite("num_groups",  &ErasureData::num_groups)
        .def_property_readonly("group_sizes", [](const ErasureData& d) {
            return std::vector<int>(d.group_sizes.begin(), d.group_sizes.begin() + d.num_groups);
        });

    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", &Chain::execute,
            pybind11::arg("board"), pybind11::arg("color_mask") = 0x0F);

    /**
     * @brief Bind Game logic
     */
    pybind11::class_<Scorer>(m, "Scorer")
        .def_static("calculateStepScore", &Scorer::calculateStepScore);

    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
        .def("get", &Tsumo::get)
        .def("setSeed", &Tsumo::setSeed)
        .def_property_readonly("seed", &Tsumo::getSeed);

    pybind11::class_<Simulator>(m, "Simulator")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
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

    /**
     * @brief Bind Puyotan Match (frame-based)
     */
    pybind11::enum_<ActionType>(m, "ActionType")
        .value("NONE", ActionType::NONE)
        .value("PASS", ActionType::PASS)
        .value("PUT", ActionType::PUT)
        .value("CHAIN", ActionType::CHAIN)
        .value("CHAIN_FALL", ActionType::CHAIN_FALL)
        .value("OJAMA", ActionType::OJAMA)
        .export_values();

    pybind11::class_<Action>(m, "Action")
        .def(pybind11::init<ActionType, int, Rotation>(),
             pybind11::arg("type") = ActionType::PASS,
             pybind11::arg("x") = 0,
             pybind11::arg("rotation") = Rotation::Up)
        .def_readwrite("type", &Action::type)
        .def_readwrite("x", &Action::x)
        .def_readwrite("rotation", &Action::rotation);

    pybind11::class_<ActionState>(m, "ActionState")
        .def_readwrite("action", &ActionState::action)
        .def_readwrite("remaining_frame", &ActionState::remaining_frame);

    pybind11::class_<PuyotanPlayer>(m, "PuyotanPlayer")
        .def_readwrite("field", &PuyotanPlayer::field)
        .def_readwrite("action_histories", &PuyotanPlayer::action_histories)
        .def_readwrite("active_next_pos", &PuyotanPlayer::active_next_pos)
        .def_readwrite("score", &PuyotanPlayer::score)
        .def_readwrite("used_score", &PuyotanPlayer::used_score)
        .def_readwrite("non_active_ojama", &PuyotanPlayer::non_active_ojama)
        .def_readwrite("active_ojama", &PuyotanPlayer::active_ojama)
        .def_readwrite("chain_count", &PuyotanPlayer::chain_count);

    pybind11::enum_<puyotan::MatchStatus>(m, "MatchStatus")
        .value("READY", puyotan::MatchStatus::READY)
        .value("PLAYING", puyotan::MatchStatus::PLAYING)
        .value("WIN_P1", puyotan::MatchStatus::WIN_P1)
        .value("WIN_P2", puyotan::MatchStatus::WIN_P2)
        .value("DRAW", puyotan::MatchStatus::DRAW)
        .export_values();

    pybind11::class_<puyotan::PuyotanMatch>(m, "PuyotanMatch")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
        .def("start", &puyotan::PuyotanMatch::start)
        .def("setAction", &puyotan::PuyotanMatch::setAction)
        .def("canStepNextFrame", &puyotan::PuyotanMatch::canStepNextFrame)
        .def("stepNextFrame", &puyotan::PuyotanMatch::stepNextFrame)
        .def("getPlayer", &puyotan::PuyotanMatch::getPlayer)
        .def("getPiece", &puyotan::PuyotanMatch::getPiece)
        .def_property_readonly("frame", &puyotan::PuyotanMatch::getFrame)
        .def_property_readonly("status", &puyotan::PuyotanMatch::getStatus)
        .def_property_readonly("status_text", &puyotan::PuyotanMatch::getStatusText);
}

} // namespace puyotan
