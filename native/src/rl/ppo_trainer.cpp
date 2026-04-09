#include <algorithm>
#include <iostream>
#include <map>
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
                             PPOConfig cfg,
                             const std::map<std::string, int>& arch_params)
    : env_(num_envs, base_seed), buffer_(num_steps, num_envs, torch::kCPU), cfg_(cfg),
      num_envs_(num_envs), num_steps_(num_steps), arch_(arch), hidden_dim_(hidden_dim),
      arch_params_(arch_params),
      episode_scores_(torch::zeros({num_envs}, torch::kFloat32)),
      episode_game_scores_(torch::zeros({num_envs}, torch::kFloat32)),
      act_p1_buf_(num_envs), act_p2_buf_(num_envs), rew_buf_(num_envs), done_buf_(num_envs),
      chain_buf_(num_envs), score_buf_(num_envs),
      obs_buf_(static_cast<std::size_t>(num_envs) * kObsBytesPerEnv),
      indices_buf_(torch::zeros({num_steps * num_envs}, torch::kInt64)),
      completed_scores_(), max_per_env_(num_envs, 0) {
    // Build policy
    if (arch == "mlp") {
        policy_ = std::make_unique<MLPPolicyWrapper>(hidden_dim);
    } else if (arch == "cnn") {
        policy_ = std::make_unique<CNNPolicyWrapper>(hidden_dim, arch_params_);
    } else if (arch == "resnet") {
        policy_ = std::make_unique<ResNetPolicyWrapper>(hidden_dim, arch_params_);
    } else {
        throw std::invalid_argument("Unknown arch: " + arch + " (expected 'mlp', 'cnn' or 'resnet')");
    }
    policy_->train(true);
    // Build optimizer
    optimizer_ = std::make_unique<torch::optim::Adam>(
        policy_->parameters(),
        torch::optim::AdamOptions(cfg_.lr).eps(1e-5));
    // Limit LibTorch threads to prevent contention on 2-core systems.
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    // Get initial observations via native path.
    env_.getObservationsNative(std::span<uint8_t>(obs_buf_));
    // [OPTIMIZATION] ZERO-COPY initialization: from_blob refers to obs_buf_ directly.
    curr_obs_ = torch::from_blob(
                    obs_buf_.data(),
                    {num_envs, kObsPlayers, kObsColors, kObsCols, kObsRows},
                    torch::kUInt8);
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
        opp_policy_ = std::make_unique<CNNPolicyWrapper>(hidden_dim_, arch_params_);
    } else if (arch_ == "resnet") {
        opp_policy_ = std::make_unique<ResNetPolicyWrapper>(hidden_dim_, arch_params_);
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
    float sum_reward = 0.0f;
    
    // [OPTIMIZATION] Re-use pre-allocated vectors to avoid heap allocation.
    completed_scores_.clear();
    std::fill(max_per_env_.begin(), max_per_env_.end(), 0);

    // Span views over the pre-allocated buffers.
    std::span<int8_t> sp_p1(act_p1_buf_);
    std::span<int8_t> sp_p2(act_p2_buf_);
    std::span<float> sp_rew(rew_buf_);
    std::span<float> sp_done(done_buf_);
    std::span<int8_t> sp_chain(chain_buf_);
    std::span<int32_t> sp_score(score_buf_);
    std::span<uint8_t> sp_obs(obs_buf_);

    // Fast PRNG state for P2 random wrapper (avoids global lock of std::rand())
    uint64_t prng_state = env_.getMatch(0).getTsumo().getSeed() + 0x123456789ULL;
    auto fast_rand = [&]() -> int8_t {
        prng_state ^= prng_state >> 12;
        prng_state ^= prng_state << 25;
        prng_state ^= prng_state >> 27;
        return (prng_state * 0x2545F4914F6CDD1DULL) % kNumActions;
    };

    for (int t = 0; t < num_steps_; ++t) {
        buffer_.storeObs(t, curr_obs_); // stored as uint8
        torch::Tensor obs_f = curr_obs_.to(torch::kFloat32);
        PolicyOutput out;
        {
            torch::NoGradGuard no_grad;
            out = policy_->getActionAndValue(obs_f);
            if (opp_policy_) {
                torch::Tensor opp_obs = obs_f.flip({1}).contiguous();
                PolicyOutput opp_out = opp_policy_->getActionAndValue(opp_obs);
                // [OPTIMIZATION] Raw pointer loop over actions tensor.
                const int64_t* opp_ptr = opp_out.actions.data_ptr<int64_t>();
                for (int i = 0; i < num_envs_; ++i) {
                    sp_p2[i] = static_cast<int8_t>(opp_ptr[i]);
                }
            }
        }
        
        // Copy P1 actions into native buffer.
        {
            // [OPTIMIZATION] Raw pointer action copy.
            const int64_t* p1_ptr = out.actions.data_ptr<int64_t>();
            for (int i = 0; i < num_envs_; ++i) {
                sp_p1[i] = static_cast<int8_t>(p1_ptr[i]);
            }
        }

        // P2 fallback (solo / random)
        if (!opp_policy_) {
            if (p2_random) {
                for (int i = 0; i < num_envs_; ++i)
                    sp_p2[i] = fast_rand(); // Lock-free PRNG
            } else {
                std::ranges::copy(sp_p1, sp_p2.begin());
            }
        }

        // ----- Native step -- OpenMP threads can run freely -----
        env_.stepNative(sp_p1, sp_p2, sp_rew, sp_done, sp_chain, sp_score, sp_obs);

        // [OPTIMIZATION] ZERO-COPY: from_blob without clone() allows RolloutBuffer to copy directly.
        auto rewards_t = torch::from_blob(sp_rew.data(), {num_envs_}, torch::kFloat32);
        auto dones_t   = torch::from_blob(sp_done.data(), {num_envs_}, torch::kFloat32);

        buffer_.storeStep(t, out.actions, out.log_probs, out.values, rewards_t, dones_t);
        // [OPTIMIZATION] Unified single loop + Raw Pointers for maximum Cache Line efficiency.
        {
            const float*   racc = sp_rew.data();
            const float*   dacc = sp_done.data();
            const int8_t*  cacc = sp_chain.data();
            const int32_t* gacc = sp_score.data();
            float*         sacc = episode_scores_.data_ptr<float>();
            float*         gsacc = episode_game_scores_.data_ptr<float>();
            int8_t*        macc = max_per_env_.data();

            for (int i = 0; i < num_envs_; ++i) {
                // 1. Accumulate stats
                sacc[i]    += racc[i];
                gsacc[i]   += static_cast<float>(gacc[i]);
                sum_reward += racc[i];
                macc[i]     = std::max(macc[i], cacc[i]);
                max_chain   = std::max(max_chain, static_cast<int>(macc[i]));

                // 2. Check done
                if (dacc[i] > 0.5f) {
                    completed_scores_.push_back(gsacc[i]); // report game score
                    sacc[i] = 0.0f;
                    gsacc[i] = 0.0f;
                }
            }
        }
        // Next observation is already in obs_buf_ (written by stepNative).
        // [OPTIMIZATION] ZERO-COPY view for the next step.
        curr_obs_ = torch::from_blob(
                        sp_obs.data(),
                        {num_envs_, kObsPlayers, kObsColors, kObsCols, kObsRows},
                        torch::kUInt8);
    }
    // Bootstrap value for GAE
    {
        torch::NoGradGuard no_grad;
        auto obs_f = curr_obs_.to(torch::kFloat32);
        auto boot = policy_->getActionAndValue(obs_f);
        buffer_.computeGae(boot.values, cfg_.gamma, cfg_.lambda);
    }
    // [OPTIMIZATION] Int accumulation prevents N float casts.
    int sum_max_int = std::accumulate(max_per_env_.begin(), max_per_env_.end(), 0);
    float sum_max_chain = static_cast<float>(sum_max_int) / static_cast<float>(num_envs_);
    float avg_reward = sum_reward / static_cast<float>(num_steps_ * num_envs_);
    
    float avg_game_score = completed_scores_.empty()
                               ? episode_game_scores_.mean().item<float>()
                               : std::accumulate(completed_scores_.begin(), completed_scores_.end(), 0.0f) / static_cast<float>(completed_scores_.size());
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
    // [OPTIMIZATION] Reuse indices_buf_ to avoid allocating a [total] tensor on the heap every epoch.
    torch::randperm_out(indices_buf_, total);

    float total_loss = 0.0f;
    for (int epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
        if (epoch > 0) {
            torch::randperm_out(indices_buf_, total); // Reshuffle for next epochs
        }
        for (int start = 0; start < total; start += cfg_.minibatch) {
            int end = std::min(start + cfg_.minibatch, total);
            auto idx = indices_buf_.slice(0, start, end);
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
