#pragma once
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace puyotan {
/**
 * @class OnnxPolicy
 * @brief High-performance AI inference engine using ONNX Runtime.
 *
 * Manages an ONNX session and provides thread-safe batch inference for
 * Puyo Puyo observation tensors.
 */
class OnnxPolicy {
  public:
    /**
     * @param model_path Path to the .onnx file.
     * @param use_cpu    true: CPU inference, false: DirectML (GPU) inference.
     */
    explicit OnnxPolicy(const std::string& model_path, bool use_cpu = true);

    /**
     * @brief Performs batch inference on a set of observations.
     * @param obs_data Pointer to raw uint8 observation data (shape [N, 2, 5, 6, 14]).
     * @param num_envs Number of parallel environment instances (N).
     * @return Vector of action indices [0-21] for each environment.
     * @note Performance: Uses DirectML or CPU optimizations via ONNX Runtime.
     */
    std::vector<int64_t> infer(const uint8_t* obs_data, int64_t num_envs) const;

    /** Whether the model is successfully loaded. */
    bool isLoaded() const {
        return session_ != nullptr;
    }

  private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Input/output names (used during inference)
    std::string input_name_{"obs"};
    std::string output_name_{"logits"};

    // Observation tensor shape: [N, 2, 5, 6, 14]
    static constexpr int64_t kPlayers = 2;
    static constexpr int64_t kColors = 5;
    static constexpr int64_t kWidth = 6;
    static constexpr int64_t kHeight = 14;
    static constexpr int64_t kObsPerEnv = kPlayers * kColors * kWidth * kHeight;
};
} // namespace puyotan
