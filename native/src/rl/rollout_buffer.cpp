#include <puyotan/rl/rollout_buffer.hpp>

#include <stdexcept>

#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs, torch::Device device)
    : num_steps_(num_steps), num_envs_(num_envs), device_(device)
{
    const auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto iopt = torch::TensorOptions().dtype(torch::kInt64).device(device);

    obs_buf_        = torch::zeros({num_steps, num_envs, kObsPlayers, kObsColors, kObsCols, kObsRows}, fopt);
    actions_buf_    = torch::zeros({num_steps, num_envs}, iopt);
    log_probs_buf_  = torch::zeros({num_steps, num_envs}, fopt);
    values_buf_     = torch::zeros({num_steps, num_envs}, fopt);
    rewards_buf_    = torch::zeros({num_steps, num_envs}, fopt);
    dones_buf_      = torch::zeros({num_steps, num_envs}, fopt);
    advantages_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    returns_buf_    = torch::zeros({num_steps, num_envs}, fopt);
}

void RolloutBuffer::storeObs(int step, const torch::Tensor& obs_uint8) {
    obs_buf_[step].copy_(obs_uint8.to(torch::kFloat32));
}

void RolloutBuffer::storeStep(int step,
                               const torch::Tensor& actions,
                               const torch::Tensor& log_probs,
                               const torch::Tensor& values,
                               const torch::Tensor& rewards,
                               const torch::Tensor& dones)
{
    actions_buf_[step]   = actions.to(device_);
    log_probs_buf_[step] = log_probs.to(device_);
    values_buf_[step]    = values.to(device_);
    rewards_buf_[step]   = rewards.to(device_);
    dones_buf_[step]     = dones.to(device_);
}

void RolloutBuffer::computeGae(const torch::Tensor& next_value,
                                float gamma, float lambda)
{
    torch::Tensor gae = torch::zeros({num_envs_}, values_buf_.options());

    for (int t = num_steps_ - 1; t >= 0; --t) {
        torch::Tensor next_val = (t == num_steps_ - 1)
            ? next_value
            : values_buf_[t + 1];

        torch::Tensor non_terminal = 1.0f - dones_buf_[t];
        torch::Tensor delta = rewards_buf_[t]
                            + gamma * next_val * non_terminal
                            - values_buf_[t];

        gae = delta + gamma * lambda * non_terminal * gae;
        advantages_buf_[t] = gae;
    }

    returns_buf_ = advantages_buf_ + values_buf_;
}

torch::Tensor RolloutBuffer::flatObs()        const { return obs_buf_.reshape({-1, kObsPlayers, kObsColors, kObsCols, kObsRows}); }
torch::Tensor RolloutBuffer::flatActions()    const { return actions_buf_.reshape({-1}); }
torch::Tensor RolloutBuffer::flatLogProbs()  const { return log_probs_buf_.reshape({-1}); }
torch::Tensor RolloutBuffer::flatValues()     const { return values_buf_.reshape({-1}); }
torch::Tensor RolloutBuffer::flatAdvantages() const { return advantages_buf_.reshape({-1}); }
torch::Tensor RolloutBuffer::flatReturns()    const { return returns_buf_.reshape({-1}); }

} // namespace puyotan::rl
