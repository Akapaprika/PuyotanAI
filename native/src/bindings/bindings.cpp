#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/beam_config_loader.hpp>
#include <puyotan/search/beam_evaluator.hpp>
#include <puyotan/search/beam_search.hpp>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace puyotan {
PYBIND11_MODULE(puyotan_native, m) {
    m.doc() = "Puyotan AI Native Engine";

    // =========================================================================
    // Common types
    // =========================================================================
    pybind11::enum_<Cell>(m, "Cell")
        .value("Red", Cell::Red)
        .value("Green", Cell::Green)
        .value("Blue", Cell::Blue)
        .value("Yellow", Cell::Yellow)
        .value("Ojama", Cell::Ojama)
        .value("Empty", Cell::Empty)
        .export_values();

    pybind11::enum_<Rotation>(m, "Rotation")
        .value("Up", Rotation::Up)
        .value("Right", Rotation::Right)
        .value("Down", Rotation::Down)
        .value("Left", Rotation::Left)
        .export_values();

    pybind11::class_<PuyoPiece>(m, "PuyoPiece")
        .def(pybind11::init<>())
        .def(pybind11::init<Cell, Cell>())
        .def_readwrite("axis", &PuyoPiece::axis)
        .def_readwrite("sub", &PuyoPiece::sub);

    // =========================================================================
    // Core
    // =========================================================================
    pybind11::class_<BitBoard>(m, "BitBoard")
        .def(pybind11::init<>())
        .def(pybind11::init<uint64_t, uint64_t>())
        .def_readwrite("lo", &BitBoard::lo)
        .def_readwrite("hi", &BitBoard::hi)
        .def("get", &BitBoard::get)
        .def("set", &BitBoard::set)
        .def("clear", &BitBoard::clear)
        .def("empty", &BitBoard::empty)
        .def("popcount", &BitBoard::popcount);

    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get", &Board::get)
        .def("set", &Board::set)
        .def("clear", &Board::clear)
        .def("placePiece", &Board::placePiece)
        .def("getBitboard", &Board::getBitboard)
        .def("getOccupied", &Board::getOccupied);

    pybind11::class_<ErasureData>(m, "ErasureData")
        .def_property_readonly("erased", [](const ErasureData& d) { return d.num_erased > 0; })
        .def_readwrite("num_erased", &ErasureData::num_erased)
        .def_readwrite("num_colors", &ErasureData::num_colors)
        .def_readwrite("num_groups", &ErasureData::num_groups)
        .def_property_readonly("group_sizes", [](const ErasureData& d) {
            return std::vector<int>(d.group_sizes.begin(), d.group_sizes.begin() + d.num_groups);
        });

    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", &Chain::execute,
                    pybind11::arg("board"), pybind11::arg("color_mask") = 0x0F);

    // =========================================================================
    // Engine
    // =========================================================================
    pybind11::class_<Scorer>(m, "Scorer")
        .def_static("calculateStepScore", &Scorer::calculateStepScore);

    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
        .def("get", &Tsumo::get)
        .def("setSeed", &Tsumo::setSeed)
        .def_property_readonly("seed", &Tsumo::getSeed);

    pybind11::enum_<ActionType>(m, "ActionType")
        .value("NONE", ActionType::None)
        .value("PASS", ActionType::Pass)
        .value("PUT", ActionType::Put)
        .value("CHAIN", ActionType::Chain)
        .value("CHAIN_FALL", ActionType::ChainFall)
        .value("OJAMA", ActionType::Ojama)
        .export_values();

    pybind11::class_<Action>(m, "Action")
        .def(pybind11::init<ActionType, int, Rotation>(),
             pybind11::arg("type") = ActionType::Pass,
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
        .def_readwrite("score", &PuyotanPlayer::score)
        .def_readwrite("used_score", &PuyotanPlayer::used_score)
        .def_readwrite("active_next_pos", &PuyotanPlayer::active_next_pos)
        .def_readwrite("non_active_ojama", &PuyotanPlayer::non_active_ojama)
        .def_readwrite("active_ojama", &PuyotanPlayer::active_ojama)
        .def_readwrite("chain_count", &PuyotanPlayer::chain_count)
        .def_readwrite("current_action", &PuyotanPlayer::current_action)
        .def_readwrite("next_action", &PuyotanPlayer::next_action);

    pybind11::enum_<MatchStatus>(m, "MatchStatus")
        .value("READY", MatchStatus::Ready)
        .value("PLAYING", MatchStatus::Playing)
        .value("WIN_P1", MatchStatus::WinP1)
        .value("WIN_P2", MatchStatus::WinP2)
        .value("DRAW", MatchStatus::Draw)
        .export_values();

    pybind11::class_<PuyotanMatch>(m, "PuyotanMatch")
        .def(pybind11::init<int32_t>(), pybind11::arg("seed") = 0)
        .def("clone", [](const PuyotanMatch& m) { return PuyotanMatch(m); })
        .def_static("runBatch", &PuyotanMatch::runBatch,
                    pybind11::arg("num_games"), pybind11::arg("seed") = 1,
                    pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("start", &PuyotanMatch::start)
        .def("setAction", &PuyotanMatch::setAction)
        .def("canStepNextFrame", &PuyotanMatch::canStepNextFrame)
        .def("stepNextFrame", &PuyotanMatch::stepNextFrame)
        .def("stepUntilDecision", &PuyotanMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("getPlayer", &PuyotanMatch::getPlayer, pybind11::return_value_policy::reference_internal)
        .def("getTsumo", &PuyotanMatch::getTsumo, pybind11::return_value_policy::reference_internal)
        .def("getPiece", &PuyotanMatch::getPiece)
        .def_property_readonly("frame", &PuyotanMatch::getFrame)
        .def_property_readonly("status", &PuyotanMatch::getStatus)
        .def("getDecisionMask", &PuyotanMatch::getDecisionMask);

    // =========================================================================
    // Environment
    // =========================================================================

    // -- RL Action Table --
    // kNumRLActions and get_rl_action() are the SINGLE SOURCE OF TRUTH for
    // the action index <-> (col, rotation) mapping used by training AND GUI.
    m.attr("kNumRLActions") = kNumRLActions;
    m.def("get_rl_action", &getRLAction, pybind11::arg("idx"),
          "Convert a flat RL action index to an Action (col, rotation). "
          "Returns Pass action for out-of-range indices.");


    // =========================================================================
    // Beam Search
    // =========================================================================
    m.def("beam_search_action",
          [](const PuyotanPlayer& player, const Tsumo& tsumo,
             const std::string& config_path,
             int beam_width,
             int look_ahead,
             bool is_solo,
             bool is_stagnated) {
              pybind11::gil_scoped_release release;

              // Load configuration with static in-memory caching
              search::BeamConfig cfg = search::BeamConfigLoader::load(config_path);

              // Override parameters if specified
              if (beam_width > 0) {
                  cfg.beam_width = beam_width;
              }
              if (look_ahead > 0) {
                  cfg.look_ahead = look_ahead;
              }

              // Apply profiles
              if (cfg.look_ahead >= 4) {
                  cfg = search::BeamConfigLoader::applyProfile(std::move(cfg), config_path, "deep_search");
              }

              if (is_solo) {
                  cfg = search::BeamConfigLoader::applyProfile(std::move(cfg), config_path, "solo_mode");
              } else {
                  cfg = search::BeamConfigLoader::applyProfile(std::move(cfg), config_path, "vs_mode");
              }

              if (is_stagnated) {
                  cfg = search::BeamConfigLoader::applyProfile(std::move(cfg), config_path, "stagnated");
              }

              return search::beamSearch(player, tsumo, cfg);
          },
          pybind11::arg("player"),
          pybind11::arg("tsumo"),
          pybind11::arg("config_path"),
          pybind11::arg("beam_width") = -1,
          pybind11::arg("look_ahead") = -1,
          pybind11::arg("is_solo") = false,
          pybind11::arg("is_stagnated") = false,
          "Run beam search internally managing config loading and profiling. Returns tuple of (RL action index, expected score).");
}
} // namespace puyotan
