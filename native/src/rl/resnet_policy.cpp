#include <puyotan/rl/resnet_policy.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace puyotan::rl {

// ---------------------------------------------------------------------------
// SEBlock (Squeeze-and-Excitation)
// ---------------------------------------------------------------------------
SEBlockImpl::SEBlockImpl(int channels, int reduction) {
    int reduced = std::max(1, channels / reduction);
    fc1 = register_module("fc1", torch::nn::Linear(channels, reduced));
    fc2 = register_module("fc2", torch::nn::Linear(reduced, channels));
}

torch::Tensor SEBlockImpl::forward(torch::Tensor x) {
    // x: [B, C, H, W]
    auto b = x.size(0);
    auto c = x.size(1);
    // Global Average Pool for channel descriptor: [B, C]
    auto y = torch::adaptive_avg_pool2d(x, {1, 1}).view({b, c});
    y = torch::relu(fc1->forward(y));
    y = torch::sigmoid(fc2->forward(y)).view({b, c, 1, 1});
    return x * y; // channel-wise recalibration
}

// ---------------------------------------------------------------------------
// ResNetBlock
// ---------------------------------------------------------------------------
ResNetBlockImpl::ResNetBlockImpl(int channels) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn1   = register_module("bn1", torch::nn::BatchNorm2d(channels));

    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn2   = register_module("bn2", torch::nn::BatchNorm2d(channels));

    se    = register_module("se", SEBlock(channels, 16));
}

torch::Tensor ResNetBlockImpl::forward(torch::Tensor x) {
    auto residual = x;

    x = torch::relu(bn1->forward(conv1->forward(x)));
    x = bn2->forward(conv2->forward(x));
    x = se->forward(x);

    x += residual;
    return torch::relu(x);
}

// ---------------------------------------------------------------------------
// ResNetBackbone  (Conv_in -> BN -> N x Block -> GAP -> FC)
// ---------------------------------------------------------------------------
ResNetBackboneImpl::ResNetBackboneImpl(int hidden_dim, int channels, int num_blocks)
    : out_dim(hidden_dim) {
    conv_in = register_module("conv_in", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(kCnnInChannels, channels, 3).padding(1).bias(false)));
    bn_in   = register_module("bn_in", torch::nn::BatchNorm2d(channels));

    blocks  = register_module("blocks", torch::nn::Sequential());
    for (int i = 0; i < num_blocks; ++i) {
        blocks->push_back(ResNetBlock(channels));
    }

    // GAP collapses [B, channels, H, W] -> [B, channels].
    // FC input is just `channels` — independent of board dimensions.
    fc = register_module("fc", torch::nn::Linear(channels, hidden_dim));

    std::cout << "[ResNetBackbone] channels=" << channels
              << "  num_blocks=" << num_blocks
              << "  GAP enabled  fc_in=" << channels << "\n";
}

torch::Tensor ResNetBackboneImpl::forward(torch::Tensor x) {
    // x: [B, 2, 5, 6, 14] float32 — guaranteed by caller
    const int64_t b = x.size(0);
    x = x.reshape({b, kCnnInChannels, kObsCols, kObsRows}); // [B, 10, 6, 14]
    x = x.transpose(2, 3).contiguous();                     // -> [B, 10, 14, 6] (Conv2d expects H,W)

    x = torch::relu(bn_in->forward(conv_in->forward(x)));   // [B, channels, 14, 6]
    x = blocks->forward(x);                                  // [B, channels, 14, 6]

    // Global Average Pooling: [B, channels, H, W] -> [B, channels]
    x = torch::adaptive_avg_pool2d(x, {1, 1}).squeeze(-1).squeeze(-1);

    return torch::relu(fc->forward(x)); // [B, hidden_dim]
}

// ---------------------------------------------------------------------------
// ResNetPolicy
// ---------------------------------------------------------------------------
ResNetPolicyImpl::ResNetPolicyImpl(int hidden_dim, int channels, int num_blocks)
    : backbone(register_module("backbone", ResNetBackbone(hidden_dim, channels, num_blocks))) {
    actor  = register_module("actor",  torch::nn::Linear(hidden_dim, kNumActions));
    critic = register_module("critic", torch::nn::Linear(hidden_dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> ResNetPolicyImpl::forward(const torch::Tensor& obs) {
    // obs is guaranteed float32 by the trainer — no redundant cast.
    auto features = backbone->forward(obs);
    return {actor->forward(features), critic->forward(features).squeeze(-1)};
}

// ---------------------------------------------------------------------------
// ResNetPolicyWrapper (IPolicy adapter)
// ---------------------------------------------------------------------------
ResNetPolicyWrapper::ResNetPolicyWrapper(int hidden_dim,
                                         const std::map<std::string, int>& arch_params) {
    auto get = [&](const std::string& key, int def) -> int {
        auto it = arch_params.find(key);
        return (it != arch_params.end()) ? it->second : def;
    };
    // Lightweight defaults for 2-core CPU; override via arch_params for GPU/cloud.
    const int channels   = get("channels",   32);
    const int num_blocks = get("num_blocks", 2);
    net_ = ResNetPolicy(hidden_dim, channels, num_blocks);
}

PolicyOutput ResNetPolicyWrapper::getActionAndValue(
    const torch::Tensor& obs,
    const torch::Tensor* provided_action) {
    auto [logits, value] = net_->forward(obs);

    auto probs         = torch::softmax(logits, -1);
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

std::vector<torch::Tensor> ResNetPolicyWrapper::parameters() {
    return net_->parameters();
}

void ResNetPolicyWrapper::train(bool mode) {
    net_->train(mode);
}

void ResNetPolicyWrapper::save(const std::string& path) {
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ofstream ofs(fspath, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("ResNetPolicyWrapper::save: cannot open: " + path);
    torch::serialize::OutputArchive archive;
    net_->save(archive);
    archive.save_to(ofs);
}

void ResNetPolicyWrapper::load(const std::string& path) {
    std::filesystem::path fspath(std::u8string(path.begin(), path.end()));
    std::ifstream ifs(fspath, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("ResNetPolicyWrapper::load: cannot open: " + path);
    torch::serialize::InputArchive archive;
    archive.load_from(ifs);
    net_->load(archive);
}

} // namespace puyotan::rl
