#include <puyotan/rl/rollout_buffer.hpp>

#include <stdexcept>

#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {
RolloutBuffer::RolloutBuffer(int num_steps, int num_envs, torch::Device device)
    : num_steps_(num_steps), num_envs_(num_envs), device_(device) {
    const auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto iopt = torch::TensorOptions().dtype(torch::kInt64).device(device);

    // Observations stored as uint8 to save memory; converted to float32 in flatObs().
    const auto u8opt = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    obs_buf_ = torch::zeros({num_steps, num_envs, kObsPlayers, kObsColors, kObsCols, kObsRows}, u8opt);
    actions_buf_ = torch::zeros({num_steps, num_envs}, iopt);
    log_probs_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    values_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    rewards_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    dones_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    advantages_buf_ = torch::zeros({num_steps, num_envs}, fopt);
    returns_buf_ = torch::zeros({num_steps, num_envs}, fopt);
}

void RolloutBuffer::storeObs(int step, const torch::Tensor& obs_uint8) {
    // Store as uint8 to avoid per-step float conversion.
    // Conversion to float32 is done once in flatObs() at update time.
    obs_buf_[step].copy_(obs_uint8);
}

void RolloutBuffer::storeStep(int step,
                              const torch::Tensor& actions,
                              const torch::Tensor& log_probs,
                              const torch::Tensor& values,
                              const torch::Tensor& rewards,
                              const torch::Tensor& dones) {
    // Tensors are already on CPU; direct copy avoids redundant device checks.
    actions_buf_[step]   = actions;
    log_probs_buf_[step] = log_probs;
    values_buf_[step]    = values;
    rewards_buf_[step]   = rewards;
    dones_buf_[step]     = dones;
}

void RolloutBuffer::computeGae(const torch::Tensor& next_value,
                               float gamma, float lambda) {
    // next_value is already a [num_envs] tensor from bootstrap.
    // We process all envs simultaneously per timestep (vectorized).
    torch::Tensor gae = torch::zeros({num_envs_}, values_buf_.options().dtype(torch::kFloat32));
    const auto rewards_f = rewards_buf_.to(torch::kFloat32);
    const auto values_f  = values_buf_.to(torch::kFloat32);
    const auto dones_f   = dones_buf_.to(torch::kFloat32);

    for (int t = num_steps_ - 1; t >= 0; --t) {
        const torch::Tensor next_val = (t == num_steps_ - 1)
                                           ? next_value
                                           : values_f[t + 1];

        const torch::Tensor non_terminal = 1.0f - dones_f[t];
        const torch::Tensor delta = rewards_f[t] + gamma * next_val * non_terminal - values_f[t];

        gae = delta + gamma * lambda * non_terminal * gae;
        advantages_buf_[t] = gae;
    }

    returns_buf_ = advantages_buf_ + values_f;
}

torch::Tensor RolloutBuffer::flatObs() const {
    // Convert uint8 -> float32 once here, for the entire buffer at update time.
    return obs_buf_.reshape({-1, kObsPlayers, kObsColors, kObsCols, kObsRows}).to(torch::kFloat32);
}
torch::Tensor RolloutBuffer::flatActions() const {
    return actions_buf_.reshape({-1});
}
torch::Tensor RolloutBuffer::flatLogProbs() const {
    return log_probs_buf_.reshape({-1});
}
torch::Tensor RolloutBuffer::flatValues() const {
    return values_buf_.reshape({-1});
}
torch::Tensor RolloutBuffer::flatAdvantages() const {
    return advantages_buf_.reshape({-1});
}
torch::Tensor RolloutBuffer::flatReturns() const {
    return returns_buf_.reshape({-1});
}
} // namespace puyotan::rl
