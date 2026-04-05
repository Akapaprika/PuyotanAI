#pragma once

#include <map>
#include <string>
#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>
#include <puyotan/rl/policy.hpp>

namespace puyotan::rl {
/**
 * @struct CNNBackboneImpl
 * @brief Two-layer CNN feature extractor with Global Average Pooling.
 *
 * Architecture: Conv2d x2 -> GAP -> Linear
 * GAP reduces [B, channels, H, W] -> [B, channels], removing the massive
 * flattened linear that was the bottleneck (5376 -> 64 units).
 */
struct CNNBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc{nullptr};
    int out_dim;

    explicit CNNBackboneImpl(int hidden_dim, int channels = 64);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CNNBackbone);

/** @struct CNNPolicyImpl */
struct CNNPolicyImpl : torch::nn::Module {
    CNNBackbone backbone;
    torch::nn::Linear actor{nullptr}, critic{nullptr};

    explicit CNNPolicyImpl(int hidden_dim = kDefaultHidden, int channels = 64);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
};
TORCH_MODULE(CNNPolicy);

/** @class CNNPolicyWrapper */
class CNNPolicyWrapper : public IPolicy {
  public:
    explicit CNNPolicyWrapper(int hidden_dim = kDefaultHidden,
                              const std::map<std::string, int>& arch_params = {});

    PolicyOutput getActionAndValue(
        const torch::Tensor& obs,
        const torch::Tensor* action = nullptr) override;

    std::vector<torch::Tensor> parameters() override;
    void train(bool mode = true) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

    CNNPolicy& module() { return net_; }

  private:
    CNNPolicy net_;
};
} // namespace puyotan::rl
