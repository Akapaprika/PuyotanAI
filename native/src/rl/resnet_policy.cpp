#include <puyotan/rl/resnet_policy.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace puyotan::rl {

// ---------------------------------------------------------------------------
// SEBlock (Squeeze-and-Excitation)
// ---------------------------------------------------------------------------
SEBlockImpl::SEBlockImpl(int channels, int reduction) {
    int reduced_channels = std::max(1, channels / reduction);
    fc1 = register_module("fc1", torch::nn::Linear(channels, reduced_channels));
    fc2 = register_module("fc2", torch::nn::Linear(reduced_channels, channels));
}

torch::Tensor SEBlockImpl::forward(torch::Tensor x) {
    // x: [B, C, H, W]
    auto b = x.size(0);
    auto c = x.size(1);
    
    // Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
    auto y = torch::adaptive_avg_pool2d(x, {1, 1}).view({b, c});
    
    y = torch::relu(fc1->forward(y));
    y = torch::sigmoid(fc2->forward(y)).view({b, c, 1, 1});
    
    // Multiply original tensor by the excitation weights
    return x * y;
}

// ---------------------------------------------------------------------------
// ResNetBlock
// ---------------------------------------------------------------------------
ResNetBlockImpl::ResNetBlockImpl(int channels) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
    
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
    
    se = register_module("se", SEBlock(channels, 16));
}

torch::Tensor ResNetBlockImpl::forward(torch::Tensor x) {
    auto residual = x;
    
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    
    x = conv2->forward(x);
    x = bn2->forward(x);
    
    x = se->forward(x);
    
    x += residual;
    x = torch::relu(x);
    
    return x;
}

// ---------------------------------------------------------------------------
// ResNetBackbone
// ---------------------------------------------------------------------------
ResNetBackboneImpl::ResNetBackboneImpl(int hidden_dim) : out_dim(hidden_dim) {
    int channels = 64; // Default starting channels for the blocks
    
    // Initial convolution mapping [B, kCnnInChannels, H, W] -> [B, channels, H, W]
    conv_in = register_module("conv_in", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(kCnnInChannels, channels, 3).padding(1).bias(false)));
    bn_in = register_module("bn_in", torch::nn::BatchNorm2d(channels));
    
    blocks = register_module("blocks", torch::nn::Sequential());
    int num_blocks = 4; // Use 4 ResNet blocks
    for (int i = 0; i < num_blocks; ++i) {
        blocks->push_back(ResNetBlock(channels));
    }
    
    // Flatten and FC
    const int conv_out = channels * kObsRows * kObsCols; // 64 * 14 * 6 = 5376
    fc = register_module("fc", torch::nn::Linear(conv_out, hidden_dim));
}

torch::Tensor ResNetBackboneImpl::forward(torch::Tensor x) {
    // x: [B, 2, 5, 6, 14] float32
    // Reshape to [B, kCnnInChannels, kObsRows, kObsCols] = [B, 10, 14, 6]
    const int64_t b = x.size(0);
    x = x.reshape({b, kCnnInChannels, kObsRows, kObsCols});
    
    x = conv_in->forward(x);
    x = bn_in->forward(x);
    x = torch::relu(x);
    
    x = blocks->forward(x);
    
    x = x.reshape({b, -1});
    x = torch::relu(fc->forward(x));
    return x;
}

// ---------------------------------------------------------------------------
// ResNetPolicy
// ---------------------------------------------------------------------------
ResNetPolicyImpl::ResNetPolicyImpl(int hidden_dim)
    : backbone(register_module("backbone", ResNetBackbone(hidden_dim))) {
    actor = register_module("actor", torch::nn::Linear(hidden_dim, kNumActions));
    critic = register_module("critic", torch::nn::Linear(hidden_dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> ResNetPolicyImpl::forward(const torch::Tensor& obs) {
    auto features = backbone->forward(obs.to(torch::kFloat32));
    return {actor->forward(features), critic->forward(features).squeeze(-1)};
}

// ---------------------------------------------------------------------------
// ResNetPolicyWrapper
// ---------------------------------------------------------------------------
ResNetPolicyWrapper::ResNetPolicyWrapper(int hidden_dim)
    : net_(hidden_dim) {}

PolicyOutput ResNetPolicyWrapper::getActionAndValue(
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
