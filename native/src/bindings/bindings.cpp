#include <puyotan/common/config.hpp>
#include <puyotan/common/types.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/vector_match.hpp>
#include <puyotan/policy/onnx_policy.hpp>
#include <puyotan/rl/constants.hpp>
#include <puyotan/rl/ppo_trainer.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

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
        .def_readwrite("last_chain_count", &PuyotanPlayer::last_chain_count)
        .def_readwrite("last_all_clear", &PuyotanPlayer::last_all_clear)
        .def_readwrite("last_erased_count", &PuyotanPlayer::last_erased_count)
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

    m.def("build_observation", [](const PuyotanMatch& match) {
            constexpr std::size_t N = ObservationBuilder::kBytesPerObservation;
            auto arr = pybind11::array_t<uint8_t>(N);
            ObservationBuilder::buildObservation(match, static_cast<uint8_t*>(arr.mutable_data()));
            return arr; }, pybind11::arg("match"));

    pybind11::class_<PuyotanVectorMatch>(m, "PuyotanVectorMatch")
        .def(pybind11::init<int, int32_t>(),
             pybind11::arg("num_matches"), pybind11::arg("base_seed") = 0)
        .def("reset", &PuyotanVectorMatch::reset, pybind11::arg("id") = -1,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("stepUntilDecision", &PuyotanVectorMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("setActions", &PuyotanVectorMatch::setActions,
             pybind11::arg("match_indices"), pybind11::arg("player_ids"), pybind11::arg("actions"))
        .def("getMatch", static_cast<PuyotanMatch& (PuyotanVectorMatch::*)(int)>(&PuyotanVectorMatch::getMatch),
             pybind11::return_value_policy::reference_internal)
        .def_property_readonly("size", &PuyotanVectorMatch::size)
        .def_readwrite("reward_calc", &PuyotanVectorMatch::reward_calc);

    // =========================================================================
    // Reward system
    // =========================================================================
    pybind11::class_<RewardWeights::Match>(m, "MatchWeights")
        .def_readwrite("win", &RewardWeights::Match::win)
        .def_readwrite("loss", &RewardWeights::Match::loss)
        .def_readwrite("draw", &RewardWeights::Match::draw);

    pybind11::class_<RewardWeights::Turn>(m, "TurnWeights")
        .def_readwrite("step_penalty", &RewardWeights::Turn::step_penalty);

    pybind11::class_<RewardWeights::Performance>(m, "PerformanceWeights")
        .def_readwrite("score_scale", &RewardWeights::Performance::score_scale)
        .def_readwrite("chain_scale", &RewardWeights::Performance::chain_scale)
        .def_readwrite("chain_power", &RewardWeights::Performance::chain_power)
        .def_readwrite("min_chain_threshold", &RewardWeights::Performance::min_chain_threshold)
        .def_readwrite("premature_chain_penalty", &RewardWeights::Performance::premature_chain_penalty)
        .def_readwrite("all_clear_bonus", &RewardWeights::Performance::all_clear_bonus)
        .def_readwrite("erasure_count_scale", &RewardWeights::Performance::erasure_count_scale)
        .def_readwrite("ojama_sent_scale", &RewardWeights::Performance::ojama_sent_scale);

    pybind11::class_<RewardWeights::Board>(m, "BoardWeights")
        .def_readwrite("puyo_count_penalty", &RewardWeights::Board::puyo_count_penalty)
        .def_readwrite("connectivity_bonus", &RewardWeights::Board::connectivity_bonus)
        .def_readwrite("isolated_puyo_penalty", &RewardWeights::Board::isolated_puyo_penalty)
        .def_readwrite("near_group_bonus", &RewardWeights::Board::near_group_bonus)
        .def_readwrite("height_variance_penalty", &RewardWeights::Board::height_variance_penalty)
        .def_readwrite("death_col_height_penalty", &RewardWeights::Board::death_col_height_penalty)
        .def_readwrite("color_diversity_reward", &RewardWeights::Board::color_diversity_reward)
        .def_readwrite("buried_puyo_penalty", &RewardWeights::Board::buried_puyo_penalty)
        .def_readwrite("ojama_drop_penalty", &RewardWeights::Board::ojama_drop_penalty)
        .def_readwrite("pending_ojama_penalty", &RewardWeights::Board::pending_ojama_penalty)
        .def_readwrite("potential_chain_bonus_scale", &RewardWeights::Board::potential_chain_bonus_scale);

    pybind11::class_<RewardWeights::Opponent>(m, "OpponentWeights")
        .def_readwrite("field_pressure_reward", &RewardWeights::Opponent::field_pressure_reward)
        .def_readwrite("connectivity_penalty", &RewardWeights::Opponent::connectivity_penalty)
        .def_readwrite("ojama_diff_scale", &RewardWeights::Opponent::ojama_diff_scale)
        .def_readwrite("initiative_bonus", &RewardWeights::Opponent::initiative_bonus);

    pybind11::class_<RewardWeights>(m, "RewardWeights")
        .def_readwrite("match", &RewardWeights::match)
        .def_readwrite("turn", &RewardWeights::turn)
        .def_readwrite("performance", &RewardWeights::performance)
        .def_readwrite("board", &RewardWeights::board)
        .def_readwrite("opponent", &RewardWeights::opponent);

    pybind11::class_<RewardCalculator>(m, "RewardCalculator")
        .def(pybind11::init<>())
        .def_readwrite("weights", &RewardCalculator::weights)
        .def("load_from_json", &RewardCalculator::load_from_json)
        .def("load_from_json_string", &RewardCalculator::load_from_json_string);

    // =========================================================================
    // OnnxPolicy
    // =========================================================================
    pybind11::class_<OnnxPolicy>(m, "OnnxPolicy")
        .def(pybind11::init<const std::string&, bool>(),
             pybind11::arg("model_path"), pybind11::arg("use_cpu") = true)
        .def("infer", [](OnnxPolicy& self, pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> obs) {
                 pybind11::gil_scoped_release release;
                 return self.infer(obs.data(), static_cast<int64_t>(obs.shape(0))); }, pybind11::arg("obs"))
        .def("is_loaded", &OnnxPolicy::isLoaded);


    // =========================================================================
    // C++ PPO Trainer
    // =========================================================================
    pybind11::class_<rl::PPOConfig>(m, "PPOConfig")
        .def(pybind11::init<>())
        .def_readwrite("gamma", &rl::PPOConfig::gamma)
        .def_readwrite("lambda_", &rl::PPOConfig::lambda)
        .def_readwrite("clip_eps", &rl::PPOConfig::clip_eps)
        .def_readwrite("entropy_coef", &rl::PPOConfig::entropy_coef)
        .def_readwrite("value_coef", &rl::PPOConfig::value_coef)
        .def_readwrite("max_grad_norm", &rl::PPOConfig::max_grad_norm)
        .def_readwrite("lr", &rl::PPOConfig::lr)
        .def_readwrite("num_epochs", &rl::PPOConfig::num_epochs)
        .def_readwrite("minibatch", &rl::PPOConfig::minibatch);

    pybind11::class_<rl::TrainMetrics>(m, "TrainMetrics")
        .def_readonly("loss", &rl::TrainMetrics::loss)
        .def_readonly("avg_reward", &rl::TrainMetrics::avg_reward)
        .def_readonly("max_chain", &rl::TrainMetrics::max_chain)
        .def_readonly("avg_max_chain", &rl::TrainMetrics::avg_max_chain)
        .def_readonly("avg_game_score", &rl::TrainMetrics::avg_game_score);

    pybind11::class_<rl::CppPPOTrainer>(m, "CppPPOTrainer")
        .def(pybind11::init<int, int, const std::string&, int, uint32_t, rl::PPOConfig>(),
             pybind11::arg("num_envs") = rl::kDefaultNumEnvs,
             pybind11::arg("num_steps") = rl::kDefaultNumSteps,
             pybind11::arg("arch") = "mlp",
             pybind11::arg("hidden_dim") = rl::kDefaultHidden,
             pybind11::arg("base_seed") = 1u,
             pybind11::arg("cfg") = rl::PPOConfig{})
        .def("trainStep", &rl::CppPPOTrainer::trainStep,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             pybind11::arg("p2_random") = false,
             "Run one rollout + PPO update. Returns TrainMetrics.")
        .def("save", &rl::CppPPOTrainer::save, pybind11::arg("path"))
        .def("load", &rl::CppPPOTrainer::load, pybind11::arg("path"))
        .def("loadP2", &rl::CppPPOTrainer::loadP2, pybind11::arg("path"))
        .def_property_readonly("env", [](rl::CppPPOTrainer& t) -> PuyotanVectorMatch& { return t.env(); }, pybind11::return_value_policy::reference_internal);
}
} // namespace puyotan
