#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/game/tsumo.hpp>
#include <puyotan/game/simulator.hpp>
#include <puyotan/game/scorer.hpp>
#include <puyotan/game/puyotan_match.hpp>
#include <puyotan/game/onnx_policy.hpp>
#include <puyotan/game/puyotan_vector_match.hpp>

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

    static auto BoardToObs = [](const Board& b) {
        // Returns a uint8 numpy array of shape [kNumColors, kWidth, kHeight]:
        //   [color_idx, col, row] -> 1 if occupied, else 0
        pybind11::array_t<uint8_t> arr({(size_t)config::Board::kNumColors, (size_t)config::Board::kWidth, (size_t)config::Board::kHeight});
        auto r = arr.mutable_unchecked<3>();
        for (int c = 0; c < config::Board::kNumColors; ++c)
            for (int x = 0; x < config::Board::kWidth; ++x)
                for (int y = 0; y < config::Board::kHeight; ++y)
                    r(c, x, y) = b.getBitboard(static_cast<Cell>(c)).get(x, y) ? 1 : 0;
        return arr;
    };

    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get", &Board::get)
        .def("set", &Board::set)
        .def("clear", &Board::clear)
        .def("placePiece", &Board::placePiece)
        .def("getBitboard", &Board::getBitboard)
        .def("getOccupied", &Board::getOccupied)
        .def("to_obs_flat", BoardToObs, pybind11::return_value_policy::move,
           "Returns a (5,6,13) float32 numpy array: one-hot color x col x row.");

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
        .def("step", &Simulator::step,
             pybind11::arg("x"), pybind11::arg("rotation"),
             "Place a piece and resolve chains. Returns delta score gained.")
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

    pybind11::class_<puyotan::PuyotanPlayer>(m, "PuyotanPlayer")
        .def_readwrite("field", &puyotan::PuyotanPlayer::field)
        .def_readwrite("score", &puyotan::PuyotanPlayer::score)
        .def_readwrite("used_score", &puyotan::PuyotanPlayer::used_score)
        .def_readwrite("active_next_pos", &puyotan::PuyotanPlayer::active_next_pos)
        .def_readwrite("non_active_ojama", &puyotan::PuyotanPlayer::non_active_ojama)
        .def_readwrite("active_ojama", &puyotan::PuyotanPlayer::active_ojama)
        .def_readwrite("chain_count", &puyotan::PuyotanPlayer::chain_count)
        .def_readwrite("current_action", &puyotan::PuyotanPlayer::current_action)
        .def_readwrite("next_action", &puyotan::PuyotanPlayer::next_action)
        .def("to_obs_flat", [](const puyotan::PuyotanPlayer& p) {
            pybind11::array_t<uint8_t> arr({(size_t)config::Board::kNumColors, (size_t)config::Board::kWidth, (size_t)config::Board::kHeight});
            auto r = arr.mutable_unchecked<3>();
            for (int c = 0; c < config::Board::kNumColors; ++c) {
                auto bb = p.field.getBitboard(static_cast<puyotan::Cell>(c));
                for (int x = 0; x < config::Board::kWidth; ++x)
                    for (int y = 0; y < config::Board::kHeight; ++y)
                        r(c, x, y) = bb.get(x, y) ? 1 : 0;
            }
            return arr;
        }, "Returns the player's field as a NumPy array [kNumColors, kWidth, kHeight]");

    pybind11::enum_<puyotan::MatchStatus>(m, "MatchStatus")
        .value("READY", puyotan::MatchStatus::READY)
        .value("PLAYING", puyotan::MatchStatus::PLAYING)
        .value("WIN_P1", puyotan::MatchStatus::WIN_P1)
        .value("WIN_P2", puyotan::MatchStatus::WIN_P2)
        .value("DRAW", puyotan::MatchStatus::DRAW)
        .export_values();

    pybind11::class_<puyotan::PuyotanMatch>(m, "PuyotanMatch")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
        .def("clone", [](const puyotan::PuyotanMatch& m) { return puyotan::PuyotanMatch(m); },
             "Returns a deep copy of this match for tree search.")
        .def_static("runBatch", &puyotan::PuyotanMatch::runBatch,
             pybind11::arg("num_games"), pybind11::arg("seed") = 1,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("start", &puyotan::PuyotanMatch::start)
        .def("setAction", &puyotan::PuyotanMatch::setAction)
        .def("canStepNextFrame", &puyotan::PuyotanMatch::canStepNextFrame)
        .def("stepNextFrame", &puyotan::PuyotanMatch::stepNextFrame)
        .def("step_until_decision", &puyotan::PuyotanMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("getPlayer", &puyotan::PuyotanMatch::getPlayer, pybind11::return_value_policy::reference_internal)
        .def("getTsumo", &puyotan::PuyotanMatch::getTsumo, pybind11::return_value_policy::reference_internal)
        .def("getPiece", &puyotan::PuyotanMatch::getPiece)
        .def_property_readonly("frame", &puyotan::PuyotanMatch::getFrame)
        .def_property_readonly("status", &puyotan::PuyotanMatch::getStatus)
        .def_property_readonly("status_text", &puyotan::PuyotanMatch::getStatusText);
    
    pybind11::class_<puyotan::PuyotanVectorMatch>(m, "PuyotanVectorMatch")
        .def(pybind11::init<int, int32_t>(), pybind11::arg("num_matches"), pybind11::arg("base_seed") = 0)
        .def("reset", &puyotan::PuyotanVectorMatch::reset, pybind11::arg("id") = -1,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("step_until_decision", &puyotan::PuyotanVectorMatch::step_until_decision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("set_actions", &puyotan::PuyotanVectorMatch::set_actions,
             pybind11::arg("match_indices"), pybind11::arg("player_ids"), pybind11::arg("actions"))
        .def("step", &puyotan::PuyotanVectorMatch::step,
             pybind11::arg("p1_actions"), pybind11::arg("p2_actions") = pybind11::none(),
             "Fast OpenMP bulk step, returning (obs, rewards, terminated)")
        .def("get_observations_all", &puyotan::PuyotanVectorMatch::get_observations_all)
        .def("get_match", &puyotan::PuyotanVectorMatch::get_match, pybind11::return_value_policy::reference_internal)
        .def_property_readonly("size", &puyotan::PuyotanVectorMatch::size);

    // ===== OnnxPolicy =====
    pybind11::class_<puyotan::OnnxPolicy>(m, "OnnxPolicy")
        .def(pybind11::init<const std::string&, bool>(),
             pybind11::arg("model_path"),
             pybind11::arg("use_cpu") = true)
        .def("infer",
             [](puyotan::OnnxPolicy& self,
                pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> obs) {
                 pybind11::gil_scoped_release release;
                 return self.infer(obs.data(), static_cast<int64_t>(obs.shape(0)));
             },
             pybind11::arg("obs"),
             "Run inference on a batch of uint8 observations. Returns list of action indices.")
        .def("is_loaded", &puyotan::OnnxPolicy::is_loaded);
}

} // namespace puyotan
