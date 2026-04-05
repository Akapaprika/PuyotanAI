#pragma once

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

/** @struct ResNetBackboneImpl */
struct ResNetBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr};
    torch::nn::Sequential blocks{nullptr};
    torch::nn::Linear fc{nullptr};
    int out_dim;

    explicit ResNetBackboneImpl(int hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNetBackbone);

/** @struct ResNetPolicyImpl */
struct ResNetPolicyImpl : torch::nn::Module {
    ResNetBackbone backbone;
    torch::nn::Linear actor{nullptr}, critic{nullptr};

    explicit ResNetPolicyImpl(int hidden_dim = kDefaultHidden);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
};
TORCH_MODULE(ResNetPolicy);

/** @class ResNetPolicyWrapper */
class ResNetPolicyWrapper : public IPolicy {
  public:
    explicit ResNetPolicyWrapper(int hidden_dim = kDefaultHidden);

    PolicyOutput getActionAndValue(
        const torch::Tensor& obs,
        const torch::Tensor* action = nullptr) override;

    std::vector<torch::Tensor> parameters() override;
    void train(bool mode = true) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

    ResNetPolicy& module() {
        return net_;
    }

  private:
    ResNetPolicy net_;
};

} // namespace puyotan::rl
