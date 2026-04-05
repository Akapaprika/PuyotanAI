#include <puyotan/rl/cnn_policy.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {
// ---------------------------------------------------------------------------
// CNNBackbone
// ---------------------------------------------------------------------------
CNNBackboneImpl::CNNBackboneImpl(int hidden_dim)
    : out_dim(hidden_dim) {
    // Layer 1: detect local adjacency patterns (3x3)
    conv1 = register_module("conv1",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(kCnnInChannels, 32, 3).padding(1)));
    // Layer 2: detect larger chain shapes
    conv2 = register_module("conv2",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));

    // After two conv layers, spatial dims unchanged (padding=1)
    // Feature map: [B, 64, kObsRows, kObsCols]
    const int conv_out = 64 * kObsRows * kObsCols; // 64 * 14 * 6 = 5376
    fc = register_module("fc", torch::nn::Linear(conv_out, hidden_dim));
}

torch::Tensor CNNBackboneImpl::forward(torch::Tensor x) {
    // x: [B, 2, 5, 6, 14] float32
    // Reshape to [B, kCnnInChannels, kObsRows, kObsCols] = [B, 10, 14, 6]
    const int64_t b = x.size(0);
    x = x.reshape({b, kCnnInChannels, kObsRows, kObsCols});

    x = torch::relu(conv1->forward(x)); // [B, 32, 14, 6]
    x = torch::relu(conv2->forward(x)); // [B, 64, 14, 6]
    x = x.reshape({b, -1});             // [B, 64*14*6]
    x = torch::relu(fc->forward(x));    // [B, hidden_dim]
    return x;
}

// ---------------------------------------------------------------------------
// CNNPolicy (Actor-Critic wrapper module)
// ---------------------------------------------------------------------------
CNNPolicyImpl::CNNPolicyImpl(int hidden_dim)
    : backbone(register_module("backbone", CNNBackbone(hidden_dim))) {
    actor = register_module("actor", torch::nn::Linear(hidden_dim, kNumActions));
    critic = register_module("critic", torch::nn::Linear(hidden_dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> CNNPolicyImpl::forward(const torch::Tensor& obs) {
    auto features = backbone->forward(obs.to(torch::kFloat32));
    return {actor->forward(features), critic->forward(features).squeeze(-1)};
}

// ---------------------------------------------------------------------------
// CNNPolicyWrapper (IPolicy adapter)
// ---------------------------------------------------------------------------
CNNPolicyWrapper::CNNPolicyWrapper(int hidden_dim)
    : net_(hidden_dim) {}

PolicyOutput CNNPolicyWrapper::getActionAndValue(
    const torch::Tensor& obs,
    const torch::Tensor* provided_action) {
    auto [logits, value] = net_->forward(obs);

    auto probs = torch::softmax(logits, -1);
    auto log_probs_all = torch::log_softmax(logits, -1);

    torch::Tensor action;
    if (provided_action != nullptr) {
        action = *provided_action;
    } else {
        action = probs.multinomial(1).squeeze(-1);
    }

    auto log_prob = log_probs_all.gather(1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy = -(probs * log_probs_all).sum(-1);

    return PolicyOutput{action, log_prob, entropy, value};
}

std::vector<torch::Tensor> CNNPolicyWrapper::parameters() {
    return net_->parameters();
}

void CNNPolicyWrapper::train(bool mode) {
    net_->train(mode);
}

void CNNPolicyWrapper::save(const std::string& path) {
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ofstream ofs(fspath, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("CNNPolicyWrapper::save: cannot open: " + path);
    torch::serialize::OutputArchive archive;
    net_->save(archive);
    archive.save_to(ofs);
}

void CNNPolicyWrapper::load(const std::string& path) {
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ifstream ifs(fspath, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("CNNPolicyWrapper::load: cannot open: " + path);
    torch::serialize::InputArchive archive;
    archive.load_from(ifs);
    net_->load(archive);
}
} // namespace puyotan::rl
