#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/search/beam_config_loader.hpp>
#include <puyotan/search/beam_evaluator.hpp>
#include <puyotan/search/beam_search.hpp>
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
        .def("get", &BitBoard::get)
        .def("set", &BitBoard::set)
        .def("clear", &BitBoard::clear)
        .def("empty", &BitBoard::empty)
        .def("popcount", &BitBoard::popcount);

    pybind11::class_<Board::ActivePuyo>(m, "ActivePuyo")
        .def_readonly("x", &Board::ActivePuyo::x)
        .def_readonly("y", &Board::ActivePuyo::y)
        .def_readonly("color", &Board::ActivePuyo::color);

    pybind11::class_<Board>(m, "Board")
        .def(pybind11::init<>())
        .def("get", &Board::get)
        .def("set", &Board::set)
        .def("clear", &Board::clear)
        .def("getActivePuyos", &Board::getActivePuyos)
        .def("get_active_puyos", &Board::getActivePuyos)
        .def("getBitboard", &Board::getBitboard)
        .def("getOccupied", &Board::getOccupied);

    pybind11::class_<ErasureData>(m, "ErasureData")
        .def_property_readonly(
            "erased", [](const ErasureData& d) { return d.num_erased > 0; })
        .def_readwrite("num_erased", &ErasureData::num_erased)
        .def_readwrite("num_colors", &ErasureData::num_colors)
        .def_readwrite("group_bonus", &ErasureData::group_bonus);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", [](Board &board) {
            return Chain::execute(board);
        });

    // =========================================================================
    // Engine
    // =========================================================================
    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("get", &Tsumo::get)
        .def_property_readonly("seed", &Tsumo::getSeed)
        .def("clone", [](const Tsumo& t) { return Tsumo(t); });

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
             pybind11::arg("type") = ActionType::Pass, pybind11::arg("x") = 0,
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
        .def("clone", [](const PuyotanPlayer& p) { return PuyotanPlayer(p); });

    pybind11::enum_<MatchStatus>(m, "MatchStatus")
        .value("READY", MatchStatus::Ready)
        .value("PLAYING", MatchStatus::Playing)
        .value("WIN_P1", MatchStatus::WinP1)
        .value("WIN_P2", MatchStatus::WinP2)
        .value("DRAW", MatchStatus::Draw)
        .export_values();

    pybind11::class_<PuyotanMatch>(m, "PuyotanMatch")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("clone", [](const PuyotanMatch& m) { return PuyotanMatch(m); })
        .def("start", &PuyotanMatch::start)
        .def("setAction", &PuyotanMatch::setAction)
        .def("canStepNextFrame", &PuyotanMatch::canStepNextFrame)
        .def("stepNextFrame", &PuyotanMatch::stepNextFrame)
        .def("stepUntilDecision", &PuyotanMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("getPlayer", &PuyotanMatch::getPlayer,
             pybind11::return_value_policy::reference_internal)
        .def("getTsumo", &PuyotanMatch::getTsumo,
             pybind11::return_value_policy::reference_internal)
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

    // BeamSearchSession (multi-turn stagnation tracking)
    pybind11::class_<search::BeamSearchSession>(m, "BeamSearchSession")
        .def(pybind11::init<>())
        .def("update", &search::BeamSearchSession::update)
        .def("isStagnated", &search::BeamSearchSession::isStagnated)
        .def("reset", &search::BeamSearchSession::reset);

    // VsEvalContext (live match state snapshot for VS evaluation)
    pybind11::class_<search::VsEvalContext>(m, "VsEvalContext")
        .def(pybind11::init<>())
        .def_readwrite("enemy_field",            &search::VsEvalContext::enemy_field)
        .def_readwrite("enemy_active_next_pos",  &search::VsEvalContext::enemy_active_next_pos)
        .def_readwrite("enemy_action_type",      &search::VsEvalContext::enemy_action_type)
        .def_readwrite("enemy_chain_count",      &search::VsEvalContext::enemy_chain_count)
        .def_readwrite("enemy_score",            &search::VsEvalContext::enemy_score)
        .def_readwrite("enemy_used_score",       &search::VsEvalContext::enemy_used_score)
        .def_readwrite("enemy_best_attack_score",&search::VsEvalContext::enemy_best_attack_score)
        .def_readwrite("enemy_prepare_turns",    &search::VsEvalContext::enemy_prepare_turns)
        .def_readwrite("enemy_best_within_4",    &search::VsEvalContext::enemy_best_within_4)
        .def_readwrite("my_best_within_4",       &search::VsEvalContext::my_best_within_4)
        .def_readwrite("enemy_active_ojama",     &search::VsEvalContext::enemy_active_ojama)
        .def_readwrite("enemy_non_active_ojama", &search::VsEvalContext::enemy_non_active_ojama)
        .def_readwrite("my_active_ojama",        &search::VsEvalContext::my_active_ojama)
        .def_readwrite("my_non_active_ojama",    &search::VsEvalContext::my_non_active_ojama);

    pybind11::class_<search::VsBeamEvalWeights>(m, "BeamEvalWeights")
        .def(pybind11::init<>())
        .def_readwrite("potential_score_scale",               &search::VsBeamEvalWeights::potential_score_scale)
        .def_readwrite("connectivity_bonus",                  &search::VsBeamEvalWeights::connectivity_bonus)
        .def_readwrite("isolated_penalty",                    &search::VsBeamEvalWeights::isolated_penalty)
        .def_readwrite("buried_penalty",                      &search::VsBeamEvalWeights::buried_penalty)
        .def_readwrite("fire_bias_permille",                  &search::VsBeamEvalWeights::fire_bias_permille)
        .def_readwrite("incoming_ojama_penalty",              &search::VsBeamEvalWeights::incoming_ojama_penalty)
        .def_readwrite("incoming_threat_bias_permille",       &search::VsBeamEvalWeights::incoming_threat_bias_permille)
        .def_readwrite("counter_attack_bias_permille",        &search::VsBeamEvalWeights::counter_attack_bias_permille)
        .def_readwrite("timing_advantage_bias_permille",      &search::VsBeamEvalWeights::timing_advantage_bias_permille)
        .def_readwrite("urgency_weight_permille",             &search::VsBeamEvalWeights::urgency_weight_permille)
        .def_readwrite("lethal_danger_scale",                 &search::VsBeamEvalWeights::lethal_danger_scale)
        .def_readwrite("effective_strike_multiplier_permille",&search::VsBeamEvalWeights::effective_strike_multiplier_permille);

    pybind11::class_<search::SoloBeamEvalWeights>(m, "SoloBeamEvalWeights")
        .def(pybind11::init<>())
        .def_readwrite("potential_score_scale", &search::SoloBeamEvalWeights::potential_score_scale);

    pybind11::class_<search::SoloBeamConfig>(m, "SoloBeamConfig")
        .def(pybind11::init<>())
        .def_readwrite("beam_width",               &search::SoloBeamConfig::beam_width)
        .def_readwrite("look_ahead",               &search::SoloBeamConfig::look_ahead)
        .def_readwrite("dbs_max_similar",          &search::SoloBeamConfig::dbs_max_similar)
        .def_readwrite("full_beam_depth",          &search::SoloBeamConfig::full_beam_depth)
        .def_readwrite("min_beam_width_ratio",     &search::SoloBeamConfig::min_beam_width_ratio)
        .def_readwrite("main_chain_threshold",     &search::SoloBeamConfig::main_chain_threshold)
        .def_readwrite("dynamic_lookahead_margin", &search::SoloBeamConfig::dynamic_lookahead_margin)
        .def_readwrite("eval_weights",             &search::SoloBeamConfig::eval_weights)
        .def("recompute_beam_widths",              &search::SoloBeamConfig::recompute_beam_widths);

    pybind11::class_<search::VsBeamConfig>(m, "VsBeamConfig")
        .def(pybind11::init<>())
        .def_readwrite("beam_width",               &search::VsBeamConfig::beam_width)
        .def_readwrite("look_ahead",               &search::VsBeamConfig::look_ahead)
        .def_readwrite("dbs_max_similar",          &search::VsBeamConfig::dbs_max_similar)
        .def_readwrite("full_beam_depth",          &search::VsBeamConfig::full_beam_depth)
        .def_readwrite("min_beam_width_ratio",     &search::VsBeamConfig::min_beam_width_ratio)
        .def_readwrite("main_chain_threshold",     &search::VsBeamConfig::main_chain_threshold)
        .def_readwrite("dynamic_lookahead_margin", &search::VsBeamConfig::dynamic_lookahead_margin)
        .def_readwrite("enable_attack_search",     &search::VsBeamConfig::enable_attack_search)
        .def_readwrite("eval_weights",             &search::VsBeamConfig::eval_weights)
        .def_readwrite("context",                  &search::VsBeamConfig::context)
        .def("recompute_beam_widths",              &search::VsBeamConfig::recompute_beam_widths);

    m.def("load_solo_config", &search::BeamConfigLoader::loadSolo, pybind11::arg("path"),
          "Load SoloBeamConfig from JSON");

    m.def("load_vs_config", &search::BeamConfigLoader::loadVs, pybind11::arg("path"),
          "Load VsBeamConfig from JSON");


    // Pure Solo beam search
    m.def(
        "solo_beam_search",
        [](const PuyotanPlayer& player, const Tsumo& tsumo,
           const search::SoloBeamConfig& cfg,
           search::BeamSearchSession* session) {
            pybind11::gil_scoped_release release;
            return search::soloBeamSearch(player, tsumo, cfg, session);
        },
        pybind11::arg("player"), pybind11::arg("tsumo"),
        pybind11::arg("cfg"), pybind11::arg("session") = nullptr,
        "Run Solo beam search with a SoloBeamConfig. Returns (RL action index, expected score).");

    m.def("get_best_leaf_field", &search::getBestLeafField,
          "Get the best leaf node board from the most recent beam search.");

    // Pure VS beam search
    m.def(
        "vs_beam_search",
        [](const PuyotanPlayer& player, const Tsumo& tsumo,
           const search::VsBeamConfig& cfg,
           search::BeamSearchSession* session) {
            pybind11::gil_scoped_release release;
            return search::vsBeamSearch(player, tsumo, cfg, session);
        },
        pybind11::arg("player"), pybind11::arg("tsumo"),
        pybind11::arg("cfg"), pybind11::arg("session") = nullptr,
        "Run VS beam search with a VsBeamConfig. Returns (RL action index, expected score).");
}
} // namespace puyotan
