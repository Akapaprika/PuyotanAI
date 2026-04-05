#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include <puyotan/env/vector_match.hpp>
#include <puyotan/rl/constants.hpp>
#include <puyotan/rl/policy.hpp>
#include <puyotan/rl/rollout_buffer.hpp>

namespace puyotan::rl {
/**
 * @struct TrainMetrics
 * @brief Metrics returned from a single trainStep() call.
 */
struct TrainMetrics {
    float loss;
    float avg_reward;     ///< Mean RL reward per step (scaled, shaped)
    int max_chain;        ///< Max chain achieved across all envs in this rollout
    float avg_max_chain;  ///< Mean of per-env max chain counts
    float avg_game_score; ///< Mean completed-episode Puyo score (raw game points, NOT RL reward)
};

/**
 * @struct PPOConfig
 * @brief All tunable PPO hyperparameters in one struct.
 */
struct PPOConfig {
    float gamma = kDefaultGamma;
    float lambda = kDefaultLambda;
    float clip_eps = kDefaultClipEps;
    float entropy_coef = kDefaultEntropyCoef;
    float value_coef = kDefaultValueCoef;
    float max_grad_norm = kDefaultMaxGradNorm;
    float lr = kDefaultLr;
    int num_epochs = kDefaultNumEpochs;
    int minibatch = kDefaultMinibatch;
};

/**
 * @class CppPPOTrainer
 * @brief Full PPO training loop in native C++.
 */
class CppPPOTrainer {
  public:
    /**
     * @brief Constructor
     * @param num_envs     Number of parallel environments.
     * @param num_steps    Rollout length per iteration.
     * @param arch         "mlp" or "cnn".
     * @param hidden_dim   Backbone hidden dimension.
     * @param base_seed    Base RNG seed for environments.
     * @param cfg          PPO hyperparameters.
     */
    CppPPOTrainer(int num_envs = kDefaultNumEnvs,
                  int num_steps = kDefaultNumSteps,
                  const std::string& arch = "mlp",
                  int hidden_dim = kDefaultHidden,
                  uint32_t base_seed = 1u,
                  PPOConfig cfg = {});

    /**
     * @brief Run one rollout + PPO update, return metrics.
     * @param p2_random  If true, P2 takes random actions (only if no frozen policy loaded).
     * @return Training metrics for this step.
     */
    TrainMetrics trainStep(bool p2_random = false);

    /**
     * @brief Save policy state_dict to file.
     * @param path Target path.
     */
    void save(const std::string& path);

    /**
     * @brief Load policy state_dict from file.
     * @param path Source path.
     */
    void load(const std::string& path);

    /**
     * @brief Load an opponent policy for self-play.
     * @param path Source path to opponent state_dict.
     */
    void loadP2(const std::string& path);

    /**
     * @brief Access to environment for configuration.
     * @return Reference to the vector match environment.
     */
    PuyotanVectorMatch& env() {
        return env_;
    }

  private:
    PuyotanVectorMatch env_;
    std::unique_ptr<IPolicy> policy_;
    std::unique_ptr<IPolicy> opp_policy_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    RolloutBuffer buffer_;
    PPOConfig cfg_;

    int num_envs_;
    int num_steps_;
    std::string arch_;
    int hidden_dim_;

    torch::Tensor curr_obs_;
    torch::Tensor episode_scores_;      ///< Accumulated RL reward per env (for avg_reward)
    torch::Tensor episode_game_scores_; ///< Accumulated raw game score per env (for avg_game_score)

    // --- Pre-allocated native I/O buffers (reused every step, no per-step allocs)
    std::vector<int> act_p1_buf_;
    std::vector<int> act_p2_buf_;
    std::vector<float> rew_buf_;
    std::vector<float> done_buf_;
    std::vector<int32_t> chain_buf_;
    std::vector<int32_t> score_buf_;
    std::vector<uint8_t> obs_buf_; ///< [N * kBytesPerObservation] flat

    /**
     * @brief Collect rollouts from environments.
     */
    std::tuple<int, float, float, float> collectRollouts_(bool p2_random);

    /**
     * @brief Compute PPO update.
     */
    float ppoUpdate_();

    /**
     * @brief Perform single minibatch update.
     */
    float updateMinibatch_(
        const torch::Tensor& b_obs,
        const torch::Tensor& b_actions,
        const torch::Tensor& b_log_probs,
        const torch::Tensor& b_advantages,
        const torch::Tensor& b_returns);

    /**
     * @brief Convert uint8 observation to float32.
     */
    static torch::Tensor obsToFloat(const torch::Tensor& obs_uint8);
};
} // namespace puyotan::rl
