#include <optional>
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
#include <puyotan/search/match_simulator.hpp>
#include <puyotan/search/negamax_search.hpp>
#include <puyotan/search/abs_search.hpp>
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
        .def_property_readonly(
            "erased", [](const ErasureData& d) { return d.num_erased > 0; })
        .def_readwrite("num_erased", &ErasureData::num_erased)
        .def_readwrite("num_colors", &ErasureData::num_colors)
        .def_readwrite("num_groups", &ErasureData::num_groups)
        .def_property_readonly("group_sizes", [](const ErasureData& d) {
            return std::vector<int>(d.group_sizes.begin(),
                                    d.group_sizes.begin() + d.num_groups);
        });

    pybind11::class_<Gravity>(m, "Gravity")
        .def_static("execute", &Gravity::execute);

    pybind11::class_<Chain>(m, "Chain")
        .def_static("execute", &Chain::execute, pybind11::arg("board"),
                    pybind11::arg("color_mask") = 0x0F);

    // =========================================================================
    // Engine
    // =========================================================================
    pybind11::class_<Scorer>(m, "Scorer")
        .def_static("calculateStepScore", &Scorer::calculateStepScore);

    pybind11::class_<Tsumo>(m, "Tsumo")
        .def(pybind11::init<uint32_t>(), pybind11::arg("seed") = 0)
        .def("get", &Tsumo::get)
        .def("setSeed", &Tsumo::setSeed)
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
        .def_static("runBatch", &PuyotanMatch::runBatch,
                    pybind11::arg("num_games"), pybind11::arg("seed") = 1,
                    pybind11::call_guard<pybind11::gil_scoped_release>())
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
        .def_readwrite("potential_score_scale",      &search::VsBeamEvalWeights::potential_score_scale)
        .def_readwrite("connectivity_bonus",         &search::VsBeamEvalWeights::connectivity_bonus)
        .def_readwrite("isolated_penalty",           &search::VsBeamEvalWeights::isolated_penalty)
        .def_readwrite("buried_penalty",             &search::VsBeamEvalWeights::buried_penalty)
        .def_readwrite("fire_bias",                  &search::VsBeamEvalWeights::fire_bias)
        .def_readwrite("incoming_ojama_penalty",     &search::VsBeamEvalWeights::incoming_ojama_penalty)
        .def_readwrite("incoming_threat_bias",       &search::VsBeamEvalWeights::incoming_threat_bias)
        .def_readwrite("counter_attack_bias",        &search::VsBeamEvalWeights::counter_attack_bias)
        .def_readwrite("timing_advantage_bias",      &search::VsBeamEvalWeights::timing_advantage_bias)
        .def_readwrite("urgency_weight",             &search::VsBeamEvalWeights::urgency_weight)
        .def_readwrite("lethal_danger_scale",        &search::VsBeamEvalWeights::lethal_danger_scale)
        .def_readwrite("effective_strike_multiplier",&search::VsBeamEvalWeights::effective_strike_multiplier);

    pybind11::class_<search::SoloBeamEvalWeights>(m, "SoloBeamEvalWeights")
        .def(pybind11::init<>())
        .def_readwrite("potential_score_scale", &search::SoloBeamEvalWeights::potential_score_scale);

    pybind11::class_<search::SoloBeamConfig>(m, "SoloBeamConfig")
        .def(pybind11::init<>())
        .def_readwrite("beam_width",            &search::SoloBeamConfig::beam_width)
        .def_readwrite("look_ahead",            &search::SoloBeamConfig::look_ahead)
        .def_readwrite("dbs_max_similar",       &search::SoloBeamConfig::dbs_max_similar)
        .def_readwrite("full_beam_depth",       &search::SoloBeamConfig::full_beam_depth)
        .def_readwrite("min_beam_width_ratio",   &search::SoloBeamConfig::min_beam_width_ratio)
        .def_readwrite("eval_weights",          &search::SoloBeamConfig::eval_weights);

    pybind11::class_<search::VsBeamConfig>(m, "VsBeamConfig")
        .def(pybind11::init<>())
        .def_readwrite("beam_width",            &search::VsBeamConfig::beam_width)
        .def_readwrite("look_ahead",            &search::VsBeamConfig::look_ahead)
        .def_readwrite("dbs_max_similar",       &search::VsBeamConfig::dbs_max_similar)
        .def_readwrite("full_beam_depth",       &search::VsBeamConfig::full_beam_depth)
        .def_readwrite("min_beam_width_ratio",   &search::VsBeamConfig::min_beam_width_ratio)
        .def_readwrite("enable_attack_search",  &search::VsBeamConfig::enable_attack_search)
        .def_readwrite("eval_weights",          &search::VsBeamConfig::eval_weights)
        .def_readwrite("context",               &search::VsBeamConfig::context);

    m.def("load_solo_config", &search::BeamConfigLoader::loadSolo, pybind11::arg("path"),
          "Load SoloBeamConfig from JSON");

    m.def("load_vs_config", &search::BeamConfigLoader::loadVs, pybind11::arg("path"),
          "Load VsBeamConfig from JSON");

    m.def("load_enemy_config", &search::BeamConfigLoader::loadEnemy, pybind11::arg("path"),
          "Load VsBeamConfig (enemy) from JSON");

    m.def(
        "beam_search_action",
        [](const PuyotanPlayer& player, const Tsumo& tsumo,
           const std::string& config_path, int beam_width, int look_ahead,
           bool is_solo, bool is_stagnated,
           const std::optional<search::VsBeamEvalWeights>& custom_weights,
           int dbs_max_similar, bool is_enemy,
           search::BeamSearchSession* session) {
            pybind11::gil_scoped_release release;

            if (is_solo) {
                search::SoloBeamConfig cfg;
                if (custom_weights.has_value()) {
                    cfg.eval_weights.potential_score_scale = custom_weights.value().potential_score_scale;
                } else {
                    cfg = search::BeamConfigLoader::loadSolo(config_path);
                }

                // Override parameters if specified
                if (beam_width > 0) { cfg.beam_width = beam_width; }
                if (look_ahead > 0) { cfg.look_ahead = look_ahead; }
                if (dbs_max_similar >= 0) { cfg.dbs_max_similar = dbs_max_similar; }

                return search::soloBeamSearch(player, tsumo, cfg, session);
            } else {
                search::VsBeamConfig cfg;
                if (custom_weights.has_value()) {
                    cfg.eval_weights = custom_weights.value();
                } else {
                    if (is_enemy) {
                        cfg = search::BeamConfigLoader::loadEnemy(config_path);
                    } else {
                        cfg = search::BeamConfigLoader::loadVs(config_path);
                    }
                }

                // Override parameters if specified
                if (beam_width > 0) { cfg.beam_width = beam_width; }
                if (look_ahead > 0) { cfg.look_ahead = look_ahead; }
                if (dbs_max_similar >= 0) { cfg.dbs_max_similar = dbs_max_similar; }

                // Apply stagnated override dynamically for VS mode
                if (is_stagnated) {
                    cfg.eval_weights.fire_bias = 0.97f;
                    cfg.eval_weights.potential_score_scale = 0.0f;
                }

                return search::vsBeamSearch(player, tsumo, cfg, session);
            }
        },
        pybind11::arg("player"), pybind11::arg("tsumo"),
        pybind11::arg("config_path"), pybind11::arg("beam_width") = -1,
        pybind11::arg("look_ahead") = -1, pybind11::arg("is_solo") = false,
        pybind11::arg("is_stagnated") = false,
        pybind11::arg("custom_weights") = std::nullopt,
        pybind11::arg("dbs_max_similar") = -1,
        pybind11::arg("is_enemy") = false,
        pybind11::arg("session") = nullptr,
        "Run beam search internally managing config loading. "
        "Returns tuple of (RL action index, expected score).");

    // Standalone VS beam search (used by VsBeamSearchAgent)
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
        "Run VS beam search with a pre-configured VsBeamConfig. "
        "Returns tuple of (RL action index, expected score).");

    // Negamax config struct
    pybind11::class_<search::NegamaxConfig>(m, "NegamaxConfig")
        .def(pybind11::init<>())
        .def_readwrite("depth",       &search::NegamaxConfig::depth)
        .def_readwrite("candidate_n", &search::NegamaxConfig::candidate_n)
        .def_readwrite("vs_config",   &search::NegamaxConfig::vs_config);

    pybind11::class_<search::NegamaxResult>(m, "NegamaxResult")
        .def(pybind11::init<>())
        .def_readwrite("best_action", &search::NegamaxResult::best_action)
        .def_readwrite("best_score",  &search::NegamaxResult::best_eval)
        .def_readwrite("candidate_evals", &search::NegamaxResult::candidate_evals);

    m.def("load_negamax_config", [](const std::string& path) {
        search::NegamaxConfig cfg;
        cfg.vs_config = search::BeamConfigLoader::loadVs(path);
        return cfg;
    }, pybind11::arg("path"), "Load NegamaxConfig from JSON");

    m.def(
        "negamax_search",
        [](const PuyotanMatch& match, int player_id, const search::NegamaxConfig& cfg) {
            pybind11::gil_scoped_release release;
            return search::negamaxSearch(match, player_id, cfg);
        },
        pybind11::arg("match"), pybind11::arg("player_id"), pybind11::arg("cfg"),
        "Run Negamax adversarial search over match states. "
        "Returns NegamaxResult with best_action and best_score.");

    pybind11::class_<search::MatchResult>(m, "MatchResult")
        .def(pybind11::init<>())
        .def_readwrite("status", &search::MatchResult::status)
        .def_readwrite("score_p1", &search::MatchResult::score_p1)
        .def_readwrite("score_p2", &search::MatchResult::score_p2)
        .def_readwrite("max_chain_p1", &search::MatchResult::max_chain_p1)
        .def_readwrite("max_chain_p2", &search::MatchResult::max_chain_p2)
        .def_readwrite("total_frames", &search::MatchResult::total_frames);

    m.def("simulate_vs_match", &search::simulateVsMatch,
          pybind11::arg("p1_cfg"), pybind11::arg("p2_cfg"), pybind11::arg("seed"), pybind11::arg("max_frames") = 15000,
          "Simulate a single VS match entirely in C++ with beam search AI. "
          "Returns MatchResult.");

    m.def("simulate_vs_matches_parallel", &search::simulateVsMatchesParallel,
          pybind11::arg("p1_cfg"), pybind11::arg("p2_cfg"), pybind11::arg("seeds"), pybind11::arg("max_frames") = 15000,
          pybind11::call_guard<pybind11::gil_scoped_release>(),
          "Simulate multiple VS matches in parallel using OpenMP. "
          "Returns list of MatchResult.");

    // ABS (Adversarial Beam Search) bindings
    pybind11::class_<search::CategoryBudgets>(m, "CategoryBudgets")
        .def(pybind11::init<>())
        .def_readwrite("build",  &search::CategoryBudgets::build)
        .def_readwrite("crush",  &search::CategoryBudgets::crush)
        .def_readwrite("strike", &search::CategoryBudgets::strike)
        .def_readwrite("evade",  &search::CategoryBudgets::evade);

    pybind11::class_<search::AbsConfig>(m, "AbsConfig")
        .def(pybind11::init<>())
        .def_readwrite("depth",                &search::AbsConfig::depth)
        .def_readwrite("chain_cutoff_enabled", &search::AbsConfig::chain_cutoff_enabled)
        .def_readwrite("my_budgets",           &search::AbsConfig::my_budgets)
        .def_readwrite("opp_budgets",          &search::AbsConfig::opp_budgets);

    pybind11::class_<search::AbsResult>(m, "AbsResult")
        .def(pybind11::init<>())
        .def_readwrite("best_action",     &search::AbsResult::best_action)
        .def_readwrite("best_eval",       &search::AbsResult::best_eval)
        .def_readwrite("candidate_evals", &search::AbsResult::candidate_evals);

    m.def("load_abs_config", [](const std::string& path) {
        search::AbsConfig cfg;
        return cfg;
    }, pybind11::arg("path"), "Load AbsConfig from JSON");

    m.def(
        "abs_search",
        [](const PuyotanMatch& match, int player_id, const search::AbsConfig& cfg) {
            pybind11::gil_scoped_release release;
            return search::absSearch(match, player_id, cfg);
        },
        pybind11::arg("match"), pybind11::arg("player_id"), pybind11::arg("cfg"),
        "Run Adversarial Beam Search over match states.");
}
} // namespace puyotan
