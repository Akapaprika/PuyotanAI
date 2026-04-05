#include <algorithm>
#include <iostream>
#include <numeric>
#include <puyotan/env/observation.hpp>
#include <puyotan/rl/cnn_policy.hpp>
#include <puyotan/rl/mlp_policy.hpp>
#include <puyotan/rl/resnet_policy.hpp>
#include <puyotan/rl/ppo_trainer.hpp>
#include <span>
#include <stdexcept>
#include <torch/torch.h>
namespace py = pybind11;
namespace puyotan::rl {
// Bytes per environment observation: [kObsPlayers * kObsColors * kObsCols * kObsRows]
static constexpr int kObsBytesPerEnv =
    kObsPlayers * kObsColors * kObsCols * kObsRows; // = 2*5*6*14 = 840
// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CppPPOTrainer::CppPPOTrainer(int num_envs, int num_steps,
                             const std::string& arch,
                             int hidden_dim,
                             uint32_t base_seed,
                             PPOConfig cfg)
    : env_(num_envs, base_seed), buffer_(num_steps, num_envs, torch::kCPU), cfg_(cfg), num_envs_(num_envs), num_steps_(num_steps), arch_(arch), hidden_dim_(hidden_dim), episode_scores_(torch::zeros({num_envs}, torch::kFloat32)), episode_game_scores_(torch::zeros({num_envs}, torch::kFloat32))
      // Pre-allocate all native I/O buffers once.
      ,
      act_p1_buf_(num_envs), act_p2_buf_(num_envs), rew_buf_(num_envs), done_buf_(num_envs), chain_buf_(num_envs), score_buf_(num_envs), obs_buf_(static_cast<std::size_t>(num_envs) * kObsBytesPerEnv) {
    // Build policy
    if (arch == "mlp") {
        policy_ = std::make_unique<MLPPolicyWrapper>(hidden_dim);
    } else if (arch == "cnn") {
        policy_ = std::make_unique<CNNPolicyWrapper>(hidden_dim);
    } else if (arch == "resnet") {
        policy_ = std::make_unique<ResNetPolicyWrapper>(hidden_dim);
    } else {
        throw std::invalid_argument("Unknown arch: " + arch + " (expected 'mlp', 'cnn' or 'resnet')");
    }
    policy_->train(true);
    // Build optimizer
    optimizer_ = std::make_unique<torch::optim::Adam>(
        policy_->parameters(),
        torch::optim::AdamOptions(cfg_.lr).eps(1e-5));
    // Get initial observations via native path.
    env_.getObservationsNative(std::span<uint8_t>(obs_buf_));
    curr_obs_ = torch::from_blob(
                    obs_buf_.data(),
                    {num_envs, kObsPlayers, kObsColors, kObsCols, kObsRows},
                    torch::kUInt8)
                    .clone();
    std::cout << "[CppPPOTrainer] arch=" << arch
              << "  hidden=" << hidden_dim
              << "  envs=" << num_envs
              << "  steps=" << num_steps << "\n";
}
// ---------------------------------------------------------------------------
// Public: trainStep
// ---------------------------------------------------------------------------
TrainMetrics CppPPOTrainer::trainStep(bool p2_random) {
    auto [max_chain, avg_max_chain, avg_reward, avg_game_score] = collectRollouts_(p2_random);
    float loss = ppoUpdate_();
    return TrainMetrics{loss, avg_reward, max_chain, avg_max_chain, avg_game_score};
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
    } else if (arch_ == "resnet") {
        opp_policy_ = std::make_unique<ResNetPolicyWrapper>(hidden_dim_);
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
    int max_chain = 0;
    float sum_max_chain = 0.0f;
    float sum_reward = 0.0f;
    std::vector<float> completed_scores;
    std::vector<int> max_per_env(num_envs_, 0);
    // Span views over the pre-allocated buffers.
    std::span<int> sp_p1(act_p1_buf_);
    std::span<int> sp_p2(act_p2_buf_);
    std::span<float> sp_rew(rew_buf_);
    std::span<float> sp_done(done_buf_);
    std::span<int32_t> sp_chain(chain_buf_);
    std::span<int32_t> sp_score(score_buf_);
    std::span<uint8_t> sp_obs(obs_buf_);
    for (int t = 0; t < num_steps_; ++t) {
        buffer_.storeObs(t, curr_obs_); // stored as uint8
        // Convert once to float32 for inference - reused for storeObs and policy
        torch::Tensor obs_f = curr_obs_.to(torch::kFloat32);
        PolicyOutput out;
        {
            torch::NoGradGuard no_grad;
            out = policy_->getActionAndValue(obs_f);
            if (opp_policy_) {
                torch::Tensor opp_obs = obs_f.flip({1}).contiguous();
                PolicyOutput opp_out = opp_policy_->getActionAndValue(opp_obs);
                // Copy opponent actions into p2 buffer.
                auto opp_acc = opp_out.actions.accessor<int64_t, 1>();
                for (int i = 0; i < num_envs_; ++i)
                    sp_p2[i] = static_cast<int>(opp_acc[i]);
            }
        }
        // Copy P1 actions into native buffer.
        {
            auto acc = out.actions.accessor<int64_t, 1>();
            for (int i = 0; i < num_envs_; ++i)
                sp_p1[i] = static_cast<int>(acc[i]);
        }
        // P2 fallback (solo / random)
        if (!opp_policy_) {
            if (p2_random) {
                for (int i = 0; i < num_envs_; ++i)
                    sp_p2[i] = std::rand() % kNumActions;
            } else {
                // Mirror P1
                std::ranges::copy(sp_p1, sp_p2.begin());
            }
        }
        // ----- Native step -- OpenMP threads can run freely -----
        env_.stepNative(sp_p1, sp_p2, sp_rew, sp_done, sp_chain, sp_score, sp_obs);
        // Build reward and done tensors directly from the native buffers (no copy).
        auto rewards_t = torch::from_blob(sp_rew.data(), {num_envs_}, torch::kFloat32).clone();
        auto dones_t = torch::from_blob(sp_done.data(), {num_envs_}, torch::kFloat32).clone();
        buffer_.storeStep(t,
                          out.actions.to(torch::kInt64),
                          out.log_probs,
                          out.values,
                          rewards_t,
                          dones_t);
        // Chain stats
        for (int i = 0; i < num_envs_; ++i) {
            int c = sp_chain[i];
            max_per_env[i] = std::max(max_per_env[i], c);
            max_chain = std::max(max_chain, c);
        }
        // Episode score accumulation (RL reward) + game score (raw points)
        {
            auto racc = rewards_t.accessor<float, 1>();
            auto dacc = dones_t.accessor<float, 1>();
            auto sacc = episode_scores_.accessor<float, 1>();
            auto gsacc = episode_game_scores_.accessor<float, 1>();
            for (int i = 0; i < num_envs_; ++i) {
                sacc[i] += racc[i];
                gsacc[i] += static_cast<float>(sp_score[i]); // raw Puyo points
                sum_reward += racc[i];
                if (dacc[i] > 0.5f) {
                    completed_scores.push_back(gsacc[i]); // report game score
                    sacc[i] = 0.0f;
                    gsacc[i] = 0.0f;
                }
            }
        }
        // Next observation is already in obs_buf_ (written by stepNative).
        curr_obs_ = torch::from_blob(
                        sp_obs.data(),
                        {num_envs_, kObsPlayers, kObsColors, kObsCols, kObsRows},
                        torch::kUInt8)
                        .clone();
    }
    // Bootstrap value for GAE
    {
        torch::NoGradGuard no_grad;
        auto obs_f = curr_obs_.to(torch::kFloat32);
        auto boot = policy_->getActionAndValue(obs_f);
        buffer_.computeGae(boot.values, cfg_.gamma, cfg_.lambda);
    }
    sum_max_chain = std::accumulate(max_per_env.begin(), max_per_env.end(), 0.0f) / static_cast<float>(num_envs_);
    float avg_reward = sum_reward / static_cast<float>(num_steps_ * num_envs_);
    // avg_game_score: use completed episode game scores;
    // fall back to current running sum mean if no episode finished this rollout.
    float avg_game_score = completed_scores.empty()
                               ? episode_game_scores_.mean().item<float>()
                               : std::accumulate(completed_scores.begin(), completed_scores.end(), 0.0f) / static_cast<float>(completed_scores.size());
    return {max_chain, sum_max_chain, avg_reward, avg_game_score};
}
// ---------------------------------------------------------------------------
// Private: ppoUpdate_
// ---------------------------------------------------------------------------
float CppPPOTrainer::ppoUpdate_() {
    policy_->train(true);
    auto b_obs = buffer_.flatObs();
    auto b_actions = buffer_.flatActions();
    auto b_log_probs = buffer_.flatLogProbs();
    auto b_advantages = buffer_.flatAdvantages();
    auto b_returns = buffer_.flatReturns();
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
    const torch::Tensor& b_returns) {
    auto out = policy_->getActionAndValue(b_obs, &b_actions);
    auto ratio = torch::exp(out.log_probs - b_log_probs);
    auto clip_r = torch::clamp(ratio, 1.0f - cfg_.clip_eps, 1.0f + cfg_.clip_eps);
    auto loss_pi = -torch::min(ratio * b_advantages, clip_r * b_advantages).mean();
    auto loss_v = 0.5f * ((out.values - b_returns).pow(2)).mean();
    auto loss_ent = -out.entropies.mean();
    auto loss = loss_pi + cfg_.value_coef * loss_v + cfg_.entropy_coef * loss_ent;
    optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(policy_->parameters(), cfg_.max_grad_norm);
    optimizer_->step();
    return loss.item<float>();
}
} // namespace puyotan::rl
