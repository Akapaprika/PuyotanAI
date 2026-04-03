#pragma once

#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {

/**
 * @class RolloutBuffer
 * @brief Pre-allocated tensor storage for PPO rollout data.
 */
class RolloutBuffer {
public:
    /**
     * @brief Constructor
     * @param num_steps  Rollout length.
     * @param num_envs   Number of parallel environments.
     * @param device     Target device (default CPU).
     */
    RolloutBuffer(int num_steps, int num_envs, torch::Device device = torch::kCPU);

    /**
     * @brief Store observation at a given timestep.
     * @param step        Timestep index.
     * @param obs_uint8   Observation tensor [N, 2, 5, 6, 14] uint8.
     */
    void storeObs(int step, const torch::Tensor& obs_uint8);

    /**
     * @brief Store scalar values for a single timestep.
     * @param step        Timestep index.
     * @param actions     Actions taken [N] int64.
     * @param log_probs   Log probabilities of actions [N] float32.
     * @param values      Value function estimates [N] float32.
     * @param rewards     Rewards received [N] float32.
     * @param dones       Done flags [N] float32 (1.0 if done).
     */
    void storeStep(int step,
                   const torch::Tensor& actions,
                   const torch::Tensor& log_probs,
                   const torch::Tensor& values,
                   const torch::Tensor& rewards,
                   const torch::Tensor& dones);

    /**
     * @brief Compute Advantage Estimates (GAE).
     * @param next_value  Value estimate for the next state [N] float32.
     * @param gamma       Discount factor.
     * @param lambda      GAE parameter.
     */
    void computeGae(const torch::Tensor& next_value,
                     float gamma = kDefaultGamma,
                     float lambda = kDefaultLambda);

    /** @brief Return flattened view of observations [S*N, ...] */
    torch::Tensor flatObs()       const;
    /** @brief Return flattened view of actions [S*N] */
    torch::Tensor flatActions()   const;
    /** @brief Return flattened view of log probabilities [S*N] */
    torch::Tensor flatLogProbs() const;
    /** @brief Return flattened view of values [S*N] */
    torch::Tensor flatValues()    const;
    /** @brief Return flattened view of advantages [S*N] */
    torch::Tensor flatAdvantages()const;
    /** @brief Return flattened view of returns (Adv + Val) [S*N] */
    torch::Tensor flatReturns()   const;

    /** @return Total number of steps in buffer (steps * envs). */
    int totalSteps() const { return num_steps_ * num_envs_; }

private:
    int num_steps_;
    int num_envs_;
    torch::Device device_;

    torch::Tensor obs_buf_;
    torch::Tensor actions_buf_;
    torch::Tensor log_probs_buf_;
    torch::Tensor values_buf_;
    torch::Tensor rewards_buf_;
    torch::Tensor dones_buf_;
    torch::Tensor advantages_buf_;
    torch::Tensor returns_buf_;
};

} // namespace puyotan::rl
