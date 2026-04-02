#include <puyotan/rl/ppo_trainer.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <puyotan/rl/mlp_policy.hpp>
#include <puyotan/rl/cnn_policy.hpp>

namespace py = pybind11;

namespace puyotan::rl {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CppPPOTrainer::CppPPOTrainer(int num_envs, int num_steps,
                              const std::string& arch,
                              int hidden_dim,
                              uint32_t base_seed,
                              PPOConfig cfg)
    : env_(num_envs, base_seed)
    , buffer_(num_steps, num_envs, torch::kCPU)
    , cfg_(cfg)
    , num_envs_(num_envs)
    , num_steps_(num_steps)
    , arch_(arch)
    , hidden_dim_(hidden_dim)
    , episode_scores_(torch::zeros({num_envs}, torch::kFloat32))
{
    // Build policy
    if (arch == "mlp") {
        policy_ = std::make_unique<MLPPolicyWrapper>(hidden_dim);
    } else if (arch == "cnn") {
        policy_ = std::make_unique<CNNPolicyWrapper>(hidden_dim);
    } else {
        throw std::invalid_argument("Unknown arch: " + arch + " (expected 'mlp' or 'cnn')");
    }
    policy_->train(true);

    // Build optimizer over all policy parameters
    optimizer_ = std::make_unique<torch::optim::Adam>(
        policy_->parameters(),
        torch::optim::AdamOptions(cfg_.lr).eps(1e-5));

    // Get initial observations
    {
        py::gil_scoped_acquire gil;
        auto obs_np = env_.getObservationsAll();
        auto* ptr   = static_cast<uint8_t*>(obs_np.mutable_data());
        curr_obs_   = torch::from_blob(ptr,
                        {num_envs, kObsPlayers, kObsColors, kObsCols, kObsRows},
                        torch::kUInt8).clone();
    }

    std::cout << "[CppPPOTrainer] arch=" << arch
              << "  hidden=" << hidden_dim
              << "  envs=" << num_envs
              << "  steps=" << num_steps << "\n";
}

// ---------------------------------------------------------------------------
// Public: trainStep
// ---------------------------------------------------------------------------
TrainMetrics CppPPOTrainer::trainStep(bool p2_random) {
    auto [max_chain, avg_max_chain, avg_reward, avg_score] = collectRollouts_(p2_random);
    float loss = ppoUpdate_();
    return TrainMetrics{loss, avg_reward, max_chain, avg_max_chain, avg_score};
}

// ---------------------------------------------------------------------------
// Public: save / load
// ---------------------------------------------------------------------------
void CppPPOTrainer::save(const std::string& path) {
    policy_->save(path);
}

void CppPPOTrainer::load(const std::string& path) {
    policy_->load(path);
}

void CppPPOTrainer::loadP2(const std::string& path) {
    if (arch_ == "mlp") {
        opp_policy_ = std::make_unique<MLPPolicyWrapper>(hidden_dim_);
    } else if (arch_ == "cnn") {
        opp_policy_ = std::make_unique<CNNPolicyWrapper>(hidden_dim_);
    } else {
        throw std::invalid_argument("Unknown arch for P2: " + arch_);
    }
    opp_policy_->load(path);
    opp_policy_->train(false);

    std::cout << "[CppPPOTrainer] Loaded P2 opponent policy from " << path << "\n";
}

// ---------------------------------------------------------------------------
// Private: collectRollouts_
// ---------------------------------------------------------------------------
std::tuple<int, float, float, float> CppPPOTrainer::collectRollouts_(bool p2_random) {
    policy_->train(false);

    int   max_chain     = 0;
    float sum_max_chain = 0.0f;
    float sum_reward    = 0.0f;
    std::vector<float> completed_scores;
    std::vector<int> max_per_env(num_envs_, 0);

    for (int t = 0; t < num_steps_; ++t) {
        buffer_.storeObs(t, curr_obs_);

        torch::Tensor obs_f = curr_obs_.to(torch::kFloat32);

        PolicyOutput out;
        torch::Tensor actions_p2;

        {
            torch::NoGradGuard no_grad;
            out = policy_->getActionAndValue(obs_f);

            if (opp_policy_) {
                torch::Tensor opp_obs = obs_f.flip({1}).contiguous();
                PolicyOutput opp_out = opp_policy_->getActionAndValue(opp_obs);
                actions_p2 = opp_out.actions;
            }
        }

        torch::Tensor actions_p1 = out.actions;

        py::gil_scoped_acquire gil;

        // p1 actions
        auto act_np = py::array_t<int>(num_envs_);
        {
            auto buf = act_np.mutable_unchecked<1>();
            auto acc = actions_p1.accessor<int64_t, 1>();
            for (int i = 0; i < num_envs_; ++i) buf(i) = static_cast<int>(acc[i]);
        }

        // p2 actions
        auto act_p2_np = py::array_t<int>(num_envs_);
        {
            auto buf = act_p2_np.mutable_unchecked<1>();
            if (opp_policy_) {
                auto opp_acc = actions_p2.accessor<int64_t, 1>();
                for (int i = 0; i < num_envs_; ++i) buf(i) = static_cast<int>(opp_acc[i]);
            } else if (p2_random) {
                for (int i = 0; i < num_envs_; ++i)
                    buf(i) = std::rand() % kNumActions;
            } else {
                auto acc = actions_p1.accessor<int64_t, 1>();
                for (int i = 0; i < num_envs_; ++i) buf(i) = static_cast<int>(acc[i]);
            }
        }

        auto result = env_.step(act_np, act_p2_np);

        auto obs_np     = result[0].cast<py::array_t<uint8_t>>();
        auto rewards_np = result[1].cast<py::array_t<float>>();
        auto dones_np   = result[2].cast<py::array_t<bool>>();
        auto chains_np  = result[3].cast<py::array_t<int32_t>>();
        auto scores_np  = result[4].cast<py::array_t<int32_t>>();

        const float* rptr = rewards_np.data();
        const bool*  dptr = dones_np.data();
        
        auto rewards_t = torch::from_blob(const_cast<float*>(rptr), {num_envs_}, torch::kFloat32).clone();
        
        auto dones_t = torch::zeros({num_envs_}, torch::kFloat32);
        auto dacc_t  = dones_t.accessor<float, 1>();
        for (int i = 0; i < num_envs_; ++i) {
            dacc_t[i] = dptr[i] ? 1.0f : 0.0f;
        }

        buffer_.storeStep(t,
            out.actions.to(torch::kInt64),
            out.log_probs,
            out.values,
            rewards_t,
            dones_t);

        {
            const int32_t* cptr = chains_np.data();
            for (int i = 0; i < num_envs_; ++i) {
                int c = cptr[i];
                max_per_env[i]  = std::max(max_per_env[i], c);
                max_chain       = std::max(max_chain, c);
            }
        }

        {
            auto racc = rewards_t.accessor<float, 1>();
            auto dacc = dones_t.accessor<float, 1>();
            auto sacc = episode_scores_.accessor<float, 1>();
            for (int i = 0; i < num_envs_; ++i) {
                sacc[i] += racc[i];
                sum_reward += racc[i];
                if (dacc[i] > 0.5f) {
                    completed_scores.push_back(sacc[i]);
                    sacc[i] = 0.0f;
                }
            }
        }

        const uint8_t* optr = obs_np.data();
        curr_obs_ = torch::from_blob(
            const_cast<uint8_t*>(optr),
            {num_envs_, kObsPlayers, kObsColors, kObsCols, kObsRows},
            torch::kUInt8).clone();
    }

    torch::Tensor next_value;
    {
        torch::NoGradGuard no_grad;
        auto obs_f = curr_obs_.to(torch::kFloat32);
        auto boot = policy_->getActionAndValue(obs_f);
        next_value = boot.values;
    }

    buffer_.computeGae(next_value, cfg_.gamma, cfg_.lambda);

    sum_max_chain = std::accumulate(max_per_env.begin(), max_per_env.end(), 0.0f)
                  / static_cast<float>(num_envs_);
    float avg_reward = sum_reward / static_cast<float>(num_steps_ * num_envs_);
    float avg_score  = completed_scores.empty()
        ? episode_scores_.mean().item<float>()
        : std::accumulate(completed_scores.begin(), completed_scores.end(), 0.0f)
          / static_cast<float>(completed_scores.size());

    return {max_chain, sum_max_chain, avg_reward, avg_score};
}

// ---------------------------------------------------------------------------
// Private: ppoUpdate_
// ---------------------------------------------------------------------------
float CppPPOTrainer::ppoUpdate_() {
    policy_->train(true);

    auto b_obs        = buffer_.flatObs();
    auto b_actions    = buffer_.flatActions();
    auto b_log_probs  = buffer_.flatLogProbs();
    auto b_advantages = buffer_.flatAdvantages();
    auto b_returns    = buffer_.flatReturns();

    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8f);

    int total = buffer_.totalSteps();
    float total_loss = 0.0f;

    for (int epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
        auto indices = torch::randperm(total, torch::kInt64);

        for (int start = 0; start < total; start += cfg_.minibatch) {
            int end = std::min(start + cfg_.minibatch, total);
            auto idx = indices.slice(0, start, end);

            total_loss += updateMinibatch_(
                b_obs.index({idx}),
                b_actions.index({idx}),
                b_log_probs.index({idx}),
                b_advantages.index({idx}),
                b_returns.index({idx}));
        }
    }

    return total_loss;
}

// ---------------------------------------------------------------------------
// Private: updateMinibatch_
// ---------------------------------------------------------------------------
float CppPPOTrainer::updateMinibatch_(
    const torch::Tensor& b_obs,
    const torch::Tensor& b_actions,
    const torch::Tensor& b_log_probs,
    const torch::Tensor& b_advantages,
    const torch::Tensor& b_returns)
{
    auto out = policy_->getActionAndValue(b_obs, &b_actions);

    auto ratio   = torch::exp(out.log_probs - b_log_probs);
    auto clip_r  = torch::clamp(ratio, 1.0f - cfg_.clip_eps, 1.0f + cfg_.clip_eps);
    auto loss_pi = -torch::min(ratio * b_advantages, clip_r * b_advantages).mean();

    auto loss_v   = 0.5f * ((out.values - b_returns).pow(2)).mean();
    auto loss_ent = -out.entropies.mean();

    auto loss = loss_pi
              + cfg_.value_coef  * loss_v
              + cfg_.entropy_coef * loss_ent;

    optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(policy_->parameters(), cfg_.max_grad_norm);
    optimizer_->step();

    return loss.item<float>();
}

} // namespace puyotan::rl
