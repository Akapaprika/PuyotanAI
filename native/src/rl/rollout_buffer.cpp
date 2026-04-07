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
    // [OPTIMIZATION] Avoid redundant .to(kFloat32) and tensor slicing.
    // Use raw pointer access for maximum vectorization efficiency.
    
    // Ensure all tensors are contiguous before taking data_ptr
    auto rew_c = rewards_buf_.contiguous();
    auto val_c = values_buf_.contiguous();
    auto done_c = dones_buf_.contiguous();
    auto adv_c = advantages_buf_.contiguous();
    auto next_val_c = next_value.contiguous(); // [num_envs_]
    
    const float* rew_ptr = rew_c.data_ptr<float>();
    const float* val_ptr = val_c.data_ptr<float>();
    const float* done_ptr = done_c.data_ptr<float>();
    const float* next_v_ptr = next_val_c.data_ptr<float>();
    float*       adv_ptr = adv_c.data_ptr<float>();
    
    // We need a small buffer to hold 'gae' for the current state (num_envs).
    // Using a std::vector avoids allocating a torch::Tensor inside the rollout collection.
    std::vector<float> curr_gae(num_envs_, 0.0f);

    for (int t = num_steps_ - 1; t >= 0; --t) {
        int offset = t * num_envs_;
        int next_offset = (t + 1) * num_envs_;
        
        for (int i = 0; i < num_envs_; ++i) {
            float nv = (t == num_steps_ - 1) ? next_v_ptr[i] : val_ptr[next_offset + i];
            float non_terminal = 1.0f - done_ptr[offset + i];
            float delta = rew_ptr[offset + i] + gamma * nv * non_terminal - val_ptr[offset + i];
            
            curr_gae[i] = delta + gamma * lambda * non_terminal * curr_gae[i];
            adv_ptr[offset + i] = curr_gae[i];
        }
    }

    // Since we called .contiguous() we must assign it back if it's a new tensor, 
    // but the underlying storage is contiguous by default for buf_.
    returns_buf_ = advantages_buf_ + values_buf_;
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
