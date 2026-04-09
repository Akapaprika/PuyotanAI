#include <puyotan/rl/cnn_policy.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include <puyotan/rl/constants.hpp>

namespace puyotan::rl {
// ---------------------------------------------------------------------------
// CNNBackbone  (Conv -> Conv -> GAP -> FC)
// ---------------------------------------------------------------------------
CNNBackboneImpl::CNNBackboneImpl(int hidden_dim, int channels)
    : out_dim(hidden_dim) {
    // Layer 1: detect local adjacency patterns (3x3)
    conv1 = register_module("conv1",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(kCnnInChannels, channels, 3).padding(1)));
    // Layer 2: detect larger chain shapes
    conv2 = register_module("conv2",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));

    // GAP collapses [B, channels, H, W] -> [B, channels] — replaces flat 5376-dim FC input.
    // Input to fc is just `channels` regardless of board size.
    fc = register_module("fc", torch::nn::Linear(channels, hidden_dim));
}

torch::Tensor CNNBackboneImpl::forward(torch::Tensor x) {
    // x: [B, 2, 5, 6, 14] float32 — already converted by caller
    const int64_t b = x.size(0);
    x = x.reshape({b, kCnnInChannels, kObsRows, kObsCols}); // [B, 10, 14, 6]

    x = torch::relu(conv1->forward(x)); // [B, channels, 14, 6]
    x = torch::relu(conv2->forward(x)); // [B, channels, 14, 6]

    // Global Average Pooling: [B, channels, H, W] -> [B, channels]
    x = torch::adaptive_avg_pool2d(x, {1, 1}).squeeze(-1).squeeze(-1);

    x = torch::relu(fc->forward(x));    // [B, hidden_dim]
    return x;
}

// ---------------------------------------------------------------------------
// CNNPolicy (Actor-Critic wrapper module)
// ---------------------------------------------------------------------------
CNNPolicyImpl::CNNPolicyImpl(int hidden_dim, int channels)
    : backbone(register_module("backbone", CNNBackbone(hidden_dim, channels))) {
    actor  = register_module("actor",  torch::nn::Linear(hidden_dim, kNumActions));
    critic = register_module("critic", torch::nn::Linear(hidden_dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> CNNPolicyImpl::forward(const torch::Tensor& obs) {
    // obs is guaranteed float32 by the trainer — no redundant cast needed.
    auto features = backbone->forward(obs);
    return {actor->forward(features), critic->forward(features).squeeze(-1)};
}

// ---------------------------------------------------------------------------
// CNNPolicyWrapper (IPolicy adapter)
// ---------------------------------------------------------------------------
CNNPolicyWrapper::CNNPolicyWrapper(int hidden_dim,
                                   const std::map<std::string, int>& arch_params) {
    auto get = [&](const std::string& key, int def) -> int {
        auto it = arch_params.find(key);
        return (it != arch_params.end()) ? it->second : def;
    };
    const int channels = get("channels", 64);
    net_ = CNNPolicy(hidden_dim, channels);
    std::cout << "[CNNPolicy] channels=" << channels
              << "  GAP enabled  fc_in=" << channels << "\n";
}

PolicyOutput CNNPolicyWrapper::getActionAndValue(
    const torch::Tensor& obs,
    const torch::Tensor* provided_action) {
    auto [logits, value] = net_->forward(obs);

    auto probs        = torch::softmax(logits, -1);
    auto log_probs_all = torch::log_softmax(logits, -1);

    torch::Tensor action;
    if (provided_action != nullptr) {
        action = *provided_action;
    } else {
        action = probs.multinomial(1).squeeze(-1);
    }

    auto log_prob = log_probs_all.gather(1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy  = -(probs * log_probs_all).sum(-1);

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
