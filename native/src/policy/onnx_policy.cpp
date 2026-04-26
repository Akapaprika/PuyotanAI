#include <puyotan/policy/onnx_policy.hpp>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace puyotan {
OnnxPolicy::OnnxPolicy(const std::string& model_path, bool use_cpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PuyotanPolicy"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try DirectML provider (if use_cpu=false and available)
    // Currently only CPU is supported (OrtSessionOptionsAppendExecutionProvider_DML() can be added later)
    (void)use_cpu; // Reserved for future DirectML support

#ifdef _WIN32
    // UTF-8 -> UTF-16 conversion for Windows
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, NULL, 0);
    std::wstring wpath(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wpath[0], size_needed);
    // Remove null terminator from wpath if present
    if (!wpath.empty() && wpath.back() == L'\0')
        wpath.pop_back();
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    input_names_[0] = input_name_.c_str();
    output_names_[0] = output_name_.c_str();

    // Resolve output action dimension once at startup to avoid per-inference shape queries.
    auto out_type_info = session_->GetOutputTypeInfo(0);
    auto out_shape = out_type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() >= 2 && out_shape[1] > 0) {
        num_actions_ = out_shape[1];
    }
}

std::vector<int64_t> OnnxPolicy::infer(const uint8_t* obs_data, int64_t num_envs) const {
    if (num_envs <= 0 || obs_data == nullptr)
        return {};

    // Build input tensor: [num_envs, 2, 5, 6, 14] (uint8)
    std::array<int64_t, 5> input_shape{num_envs, kPlayers, kColors, kWidth, kHeight};
    const int64_t total_elems = num_envs * kObsPerEnv;

    // Use uint8 data directly as a tensor (specify type explicitly to avoid float misinterpretation)
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        memory_info_,
        const_cast<uint8_t*>(obs_data), total_elems * sizeof(uint8_t),
        input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Run inference
    auto output_tensors = session_->Run(
        run_options_,
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1);

    // logits [num_envs, 22] -> argmax -> action index
    const float* logits = output_tensors[0].GetTensorData<float>();

    std::vector<int64_t> actions(num_envs);
    for (int64_t i = 0; i < num_envs; ++i) {
        const float* row = logits + i * num_actions_;
        int64_t best = 0;
        float best_logit = row[0];
        for (int64_t a = 1; a < num_actions_; ++a) {
            if (row[a] > best_logit) {
                best_logit = row[a];
                best = a;
            }
        }
        actions[i] = best;
    }
    return actions;
}
} // namespace puyotan
