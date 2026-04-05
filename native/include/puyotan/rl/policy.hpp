#pragma once

#include <string>
#include <torch/torch.h>
#include <vector>

namespace puyotan::rl {
/**
 * @struct PolicyOutput
 * @brief Result of a single forward + sample pass through the policy.
 */
struct PolicyOutput {
    torch::Tensor actions;
    torch::Tensor log_probs;
    torch::Tensor entropies;
    torch::Tensor values;
};

/**
 * @class IPolicy
 * @brief Abstract Actor-Critic policy interface.
 */
class IPolicy {
  public:
    virtual ~IPolicy() = default;

    /**
     * @brief Forward pass: sample or evaluate action(s).
     * @param obs   float32 tensor [batch, 2, 5, 6, 14]
     * @param action Optional: if provided, evaluate given actions rather than sampling.
     * @return PolicyOutput containing actions, log_probs, entropies, values.
     */
    virtual PolicyOutput getActionAndValue(
        const torch::Tensor& obs,
        const torch::Tensor* action = nullptr) = 0;

    /**
     * @brief Return all trainable parameters for the optimizer.
     */
    virtual std::vector<torch::Tensor> parameters() = 0;

    /**
     * @brief Switch between training and inference mode (affects dropout etc.).
     */
    virtual void train(bool mode = true) = 0;

    /**
     * @brief Persist the policy state to a file.
     */
    virtual void save(const std::string& path) = 0;

    /**
     * @brief Restore the policy state from a file.
     */
    virtual void load(const std::string& path) = 0;
};
} // namespace puyotan::rl
