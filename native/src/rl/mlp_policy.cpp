#include <filesystem>
#include <fstream>
#include <puyotan/rl/constants.hpp>
#include <puyotan/rl/mlp_policy.hpp>
#include <stdexcept>
#include <torch/torch.h>
namespace puyotan::rl {
// ---------------------------------------------------------------------------
// MLPBackbone
// ---------------------------------------------------------------------------
MLPBackboneImpl::MLPBackboneImpl(int hidden_dim)
    : out_dim(hidden_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(kObsFlatSize, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
}
torch::Tensor MLPBackboneImpl::forward(torch::Tensor x) {
    // x: [batch, 2, 5, 6, 14] float32 -> flatten to [batch, 840]
    x = x.reshape({x.size(0), kObsFlatSize});
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return x; // [batch, hidden_dim]
}
// ---------------------------------------------------------------------------
// MLPPolicy (Actor-Critic wrapper module)
// ---------------------------------------------------------------------------
MLPPolicyImpl::MLPPolicyImpl(int hidden_dim)
    : backbone(register_module("backbone", MLPBackbone(hidden_dim))) {
    actor = register_module("actor", torch::nn::Linear(hidden_dim, kNumActions));
    critic = register_module("critic", torch::nn::Linear(hidden_dim, 1));
}
std::pair<torch::Tensor, torch::Tensor> MLPPolicyImpl::forward(const torch::Tensor& obs) {
    auto features = backbone->forward(obs.to(torch::kFloat32));
    return {actor->forward(features), critic->forward(features).squeeze(-1)};
}
// ---------------------------------------------------------------------------
// MLPPolicyWrapper (IPolicy adapter)
// ---------------------------------------------------------------------------
MLPPolicyWrapper::MLPPolicyWrapper(int hidden_dim)
    : net_(hidden_dim) {}
PolicyOutput MLPPolicyWrapper::getActionAndValue(
    const torch::Tensor& obs,
    const torch::Tensor* provided_action) {
    auto [logits, value] = net_->forward(obs);
    // Categorical distribution
    auto probs = torch::softmax(logits, /*dim=*/-1);
    auto log_probs_all = torch::log_softmax(logits, -1);
    torch::Tensor action;
    if (provided_action != nullptr) {
        action = *provided_action;
    } else {
        action = probs.multinomial(1).squeeze(-1); // [batch] int64
    }
    auto log_prob = log_probs_all.gather(1, action.unsqueeze(-1)).squeeze(-1);
    // Entropy: -sum(p * log p)
    auto entropy = -(probs * log_probs_all).sum(-1);
    return PolicyOutput{action, log_prob, entropy, value};
}
std::vector<torch::Tensor> MLPPolicyWrapper::parameters() {
    return net_->parameters();
}
void MLPPolicyWrapper::train(bool mode) {
    net_->train(mode);
}
void MLPPolicyWrapper::save(const std::string& path) {
    // Use std::filesystem::path to correctly handle UTF-8 paths on Windows.
    // LibTorch's archive.save_to(path) fails on non-ASCII paths with MSVC.
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ofstream ofs(fspath, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("MLPPolicyWrapper::save: cannot open: " + path);
    torch::serialize::OutputArchive archive;
    net_->save(archive);
    archive.save_to(ofs);
}
void MLPPolicyWrapper::load(const std::string& path) {
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ifstream ifs(fspath, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("MLPPolicyWrapper::load: cannot open: " + path);
    torch::serialize::InputArchive archive;
    archive.load_from(ifs);
    net_->load(archive);
}
} // namespace puyotan::rl
