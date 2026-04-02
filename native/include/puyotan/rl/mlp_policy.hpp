#pragma once

#include <string>
#include <torch/torch.h>

#include <puyotan/rl/policy.hpp>
#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {

/** @struct MLPBackboneImpl */
struct MLPBackboneImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int out_dim;

    explicit MLPBackboneImpl(int hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MLPBackbone);

/** @struct MLPPolicyImpl */
struct MLPPolicyImpl : torch::nn::Module {
    MLPBackbone backbone;
    torch::nn::Linear actor{nullptr}, critic{nullptr};

    explicit MLPPolicyImpl(int hidden_dim = kDefaultHidden);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
};
TORCH_MODULE(MLPPolicy);

/** @class MLPPolicyWrapper */
class MLPPolicyWrapper : public IPolicy {
public:
    explicit MLPPolicyWrapper(int hidden_dim = kDefaultHidden);

    PolicyOutput getActionAndValue(
        const torch::Tensor& obs,
        const torch::Tensor* action = nullptr) override;

    std::vector<torch::Tensor> parameters() override;
    void train(bool mode = true) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

    MLPPolicy& module() { return net_; }

private:
    MLPPolicy net_;
};

} // namespace puyotan::rl
