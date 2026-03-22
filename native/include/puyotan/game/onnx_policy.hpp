#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace puyotan {

/**
 * AI inference class using ONNX Runtime (Single Responsibility).
 *
 * Usage:
 *   OnnxPolicy policy("models/puyotan.onnx");
 *   auto actions = policy.infer(obs_uint8, num_envs);
 */
class OnnxPolicy {
public:
    /**
     * @param model_path Path to the .onnx file.
     * @param use_cpu    true: CPU inference, false: DirectML (GPU) inference.
     */
    explicit OnnxPolicy(const std::string& model_path, bool use_cpu = true);

    /**
     * Batch inference: Returns action indices from uint8 observation data.
     *
     * @param obs_data  Observation data (num_envs * 2 * 5 * 6 * 13 bytes, uint8).
     * @param num_envs  Number of parallel environments.
     * @return          Action indices for each environment [0, 21].
     */
    std::vector<int64_t> infer(const uint8_t* obs_data, int64_t num_envs);

    /** Whether the model is successfully loaded. */
    bool is_loaded() const { return session_ != nullptr; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // 入力/出力名（推論時に使用）
    std::string input_name_{"obs"};
    std::string output_name_{"logits"};

    // 観測テンソルの形状: [N, 2, 5, 6, 13]
    static constexpr int64_t kPlayers = 2;
    static constexpr int64_t kColors  = 5;
    static constexpr int64_t kWidth   = 6;
    static constexpr int64_t kHeight  = 13;
    static constexpr int64_t kObsPerEnv = kPlayers * kColors * kWidth * kHeight;
};

} // namespace puyotan
