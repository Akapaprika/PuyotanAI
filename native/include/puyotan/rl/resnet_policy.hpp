#pragma once

#include <map>
#include <string>
#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>
#include <puyotan/rl/policy.hpp>

namespace puyotan::rl {

/** @struct SEBlockImpl */
struct SEBlockImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    explicit SEBlockImpl(int channels, int reduction = 16);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SEBlock);

/** @struct ResNetBlockImpl */
struct ResNetBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    SEBlock se{nullptr};

    explicit ResNetBlockImpl(int channels);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNetBlock);

/**
 * @struct ResNetBackboneImpl
 * @brief ResNet feature extractor with Global Average Pooling.
 *
 * Architecture: Conv_in -> BN -> ReLU -> N x ResNetBlock -> GAP -> Linear
 * GAP reduces [B, channels, H, W] -> [B, channels], dramatically cutting
 * the FC layer size and making this viable on a 2-core CPU.
 *
 * Default light config (2-core): channels=32, num_blocks=2
 * Heavy config (GPU/cloud):      channels=64, num_blocks=4+
 */
struct ResNetBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr};
    torch::nn::Sequential blocks{nullptr};
    torch::nn::Linear fc{nullptr};
    int out_dim;

    explicit ResNetBackboneImpl(int hidden_dim, int channels = 32, int num_blocks = 2);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNetBackbone);

/** @struct ResNetPolicyImpl */
struct ResNetPolicyImpl : torch::nn::Module {
    ResNetBackbone backbone;
    torch::nn::Linear actor{nullptr}, critic{nullptr};

    explicit ResNetPolicyImpl(int hidden_dim = kDefaultHidden, int channels = 32, int num_blocks = 2);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
};
TORCH_MODULE(ResNetPolicy);

/** @class ResNetPolicyWrapper */
class ResNetPolicyWrapper : public IPolicy {
  public:
    explicit ResNetPolicyWrapper(int hidden_dim = kDefaultHidden,
                                 const std::map<std::string, int>& arch_params = {});

    PolicyOutput getActionAndValue(
        const torch::Tensor& obs,
        const torch::Tensor* action = nullptr) override;

    std::vector<torch::Tensor> parameters() override;
    void train(bool mode = true) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

    ResNetPolicy& module() { return net_; }

  private:
    ResNetPolicy net_;
};

} // namespace puyotan::rl
