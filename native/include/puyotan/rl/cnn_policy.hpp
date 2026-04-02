#pragma once

#include <string>
#include <torch/torch.h>

#include <puyotan/rl/policy.hpp>
#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {

/** @struct CNNBackboneImpl */
struct CNNBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc{nullptr};
    int out_dim;

    explicit CNNBackboneImpl(int hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CNNBackbone);

/** @struct CNNPolicyImpl */
struct CNNPolicyImpl : torch::nn::Module {
    CNNBackbone backbone;
    torch::nn::Linear actor{nullptr}, critic{nullptr};

    explicit CNNPolicyImpl(int hidden_dim = kDefaultHidden);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
};
TORCH_MODULE(CNNPolicy);

/** @class CNNPolicyWrapper */
class CNNPolicyWrapper : public IPolicy {
public:
    explicit CNNPolicyWrapper(int hidden_dim = kDefaultHidden);

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
