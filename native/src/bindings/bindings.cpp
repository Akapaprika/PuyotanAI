#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <puyotan/common/types.hpp>
#include <puyotan/common/config.hpp>
#include <puyotan/core/board.hpp>
#include <puyotan/core/gravity.hpp>
#include <puyotan/core/chain.hpp>
#include <puyotan/engine/tsumo.hpp>
#include <puyotan/engine/scorer.hpp>
#include <puyotan/engine/match.hpp>
#include <puyotan/policy/onnx_policy.hpp>
#include <puyotan/env/vector_match.hpp>
#include <puyotan/env/observation.hpp>
#include <puyotan/env/rl_utils.hpp>

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

    /**
     * @brief Bind Puyotan Match (frame-based)
     */
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

    pybind11::class_<puyotan::PuyotanPlayer>(m, "PuyotanPlayer")
        .def_readwrite("field", &puyotan::PuyotanPlayer::field)
        .def_readwrite("score", &puyotan::PuyotanPlayer::score)
        .def_readwrite("used_score", &puyotan::PuyotanPlayer::used_score)
        .def_readwrite("active_next_pos", &puyotan::PuyotanPlayer::active_next_pos)
        .def_readwrite("non_active_ojama", &puyotan::PuyotanPlayer::non_active_ojama)
        .def_readwrite("active_ojama", &puyotan::PuyotanPlayer::active_ojama)
        .def_readwrite("chain_count", &puyotan::PuyotanPlayer::chain_count)
        .def_readwrite("last_chain_count", &puyotan::PuyotanPlayer::last_chain_count)
        .def_readwrite("current_action", &puyotan::PuyotanPlayer::current_action)
        .def_readwrite("next_action", &puyotan::PuyotanPlayer::next_action);

    pybind11::enum_<puyotan::MatchStatus>(m, "MatchStatus")
        .value("READY", puyotan::MatchStatus::Ready)
        .value("PLAYING", puyotan::MatchStatus::Playing)
        .value("WIN_P1", puyotan::MatchStatus::WinP1)
        .value("WIN_P2", puyotan::MatchStatus::WinP2)
        .value("DRAW", puyotan::MatchStatus::Draw)
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
        .def("stepUntilDecision", &puyotan::PuyotanMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("getPlayer", &puyotan::PuyotanMatch::getPlayer, pybind11::return_value_policy::reference_internal)
        .def("getTsumo", &puyotan::PuyotanMatch::getTsumo, pybind11::return_value_policy::reference_internal)
        .def("getPiece", &puyotan::PuyotanMatch::getPiece)
        .def_property_readonly("frame", &puyotan::PuyotanMatch::getFrame)
        .def_property_readonly("status", &puyotan::PuyotanMatch::getStatus)
        .def("getDecisionMask", &puyotan::PuyotanMatch::getDecisionMask,
             "Returns bitmask of players needing a PUT decision (0=auto, 1=P1, 2=P2, 3=both)");

    m.def("compute_gae", &puyotan::computeGae,
          pybind11::arg("rewards"), pybind11::arg("values"),
          pybind11::arg("dones"), pybind11::arg("next_value"),
          pybind11::arg("gamma"), pybind11::arg("lam"),
          "Computes GAE entirely in C++ (OpenMP+SIMD)");

    // Free helper: build a single flat observation buffer from a live PuyotanMatch.
    // Returns a new numpy array of shape (kBytesPerObservation,) dtype=uint8.
    m.def("build_observation",
        [](const puyotan::PuyotanMatch& match) {
            constexpr std::size_t N = puyotan::ObservationBuilder::kBytesPerObservation;
            auto arr = pybind11::array_t<uint8_t>(N);
            puyotan::ObservationBuilder::buildObservation(
                match, static_cast<uint8_t*>(arr.mutable_data()));
            return arr;
        },
        pybind11::arg("match"),
        "Build a flat uint8 observation array (2-player) from a live match.");

    
    pybind11::class_<puyotan::PuyotanVectorMatch>(m, "PuyotanVectorMatch")
        .def(pybind11::init<int, int32_t>(), pybind11::arg("num_matches"), pybind11::arg("base_seed") = 0)
        .def("reset", &puyotan::PuyotanVectorMatch::reset, pybind11::arg("id") = -1,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("stepUntilDecision", &puyotan::PuyotanVectorMatch::stepUntilDecision,
             pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("setActions", &puyotan::PuyotanVectorMatch::setActions,
             pybind11::arg("match_indices"), pybind11::arg("player_ids"), pybind11::arg("actions"))
        .def("step", &puyotan::PuyotanVectorMatch::step,
             pybind11::arg("p1_actions"), pybind11::arg("p2_actions") = pybind11::none(), pybind11::arg("out_obs") = pybind11::none(),
             "Fast OpenMP bulk step, returning (obs, rewards, terminated)")
        .def("getObservationsAll", &puyotan::PuyotanVectorMatch::getObservationsAll, pybind11::arg("out_obs") = pybind11::none())
        .def("getMatch", static_cast<puyotan::PuyotanMatch& (puyotan::PuyotanVectorMatch::*)(int)>(&puyotan::PuyotanVectorMatch::getMatch), pybind11::return_value_policy::reference_internal)
        .def_property_readonly("size", &puyotan::PuyotanVectorMatch::size)
        .def_readwrite("reward_calc", &puyotan::PuyotanVectorMatch::reward_calc);

    // ===== Reward System =====
    pybind11::class_<puyotan::RewardWeights::Match>(m, "MatchWeights")
        .def_readwrite("win",  &puyotan::RewardWeights::Match::win)
        .def_readwrite("loss", &puyotan::RewardWeights::Match::loss)
        .def_readwrite("draw", &puyotan::RewardWeights::Match::draw);

    pybind11::class_<puyotan::RewardWeights::Turn>(m, "TurnWeights")
        .def_readwrite("step_penalty", &puyotan::RewardWeights::Turn::step_penalty);

    pybind11::class_<puyotan::RewardWeights::Performance>(m, "PerformanceWeights")
        .def_readwrite("score_scale",       &puyotan::RewardWeights::Performance::score_scale)
        .def_readwrite("chain_bonus_scale", &puyotan::RewardWeights::Performance::chain_bonus_scale);

    pybind11::class_<puyotan::RewardWeights::Board>(m, "BoardWeights")
        .def_readwrite("puyo_count_penalty", &puyotan::RewardWeights::Board::puyo_count_penalty)
        .def_readwrite("connectivity_bonus", &puyotan::RewardWeights::Board::connectivity_bonus)
        .def_readwrite("isolated_puyo_penalty",      &puyotan::RewardWeights::Board::isolated_puyo_penalty)
        .def_readwrite("death_col_height_penalty",   &puyotan::RewardWeights::Board::death_col_height_penalty)
        .def_readwrite("color_diversity_reward",     &puyotan::RewardWeights::Board::color_diversity_reward)
        .def_readwrite("buried_puyo_penalty", &puyotan::RewardWeights::Board::buried_puyo_penalty)
        .def_readwrite("ojama_drop_penalty",  &puyotan::RewardWeights::Board::ojama_drop_penalty)
        .def_readwrite("potential_chain_bonus_scale", &puyotan::RewardWeights::Board::potential_chain_bonus_scale);

    pybind11::class_<puyotan::RewardWeights::Opponent>(m, "OpponentWeights")
        .def_readwrite("field_pressure_reward", &puyotan::RewardWeights::Opponent::field_pressure_reward)
        .def_readwrite("connectivity_penalty",  &puyotan::RewardWeights::Opponent::connectivity_penalty)
        .def_readwrite("ojama_diff_scale",      &puyotan::RewardWeights::Opponent::ojama_diff_scale)
        .def_readwrite("initiative_bonus",      &puyotan::RewardWeights::Opponent::initiative_bonus);

    pybind11::class_<puyotan::RewardWeights>(m, "RewardWeights")
        .def_readwrite("match",       &puyotan::RewardWeights::match)
        .def_readwrite("turn",        &puyotan::RewardWeights::turn)
        .def_readwrite("performance", &puyotan::RewardWeights::performance)
        .def_readwrite("board",       &puyotan::RewardWeights::board)
        .def_readwrite("opponent",    &puyotan::RewardWeights::opponent);

    pybind11::class_<puyotan::RewardCalculator>(m, "RewardCalculator")
        .def(pybind11::init<>())
        .def_readwrite("weights", &puyotan::RewardCalculator::weights)
        .def("load_from_json", &puyotan::RewardCalculator::load_from_json)
        .def("load_from_json_string", &puyotan::RewardCalculator::load_from_json_string);

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
        .def("is_loaded", &puyotan::OnnxPolicy::isLoaded);
}
}

} // namespace puyotan
