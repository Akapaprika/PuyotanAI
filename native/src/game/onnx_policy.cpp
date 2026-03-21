#include "puyotan/game/onnx_policy.hpp"
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace puyotan {

OnnxPolicy::OnnxPolicy(const std::string& model_path, bool use_cpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PuyotanPolicy")
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // DirectML プロバイダーを試みる（use_cpu=false かつ利用可能な場合）
    // 現時点では CPU のみサポート（将来的に OrtSessionOptionsAppendExecutionProvider_DML() を追加可能）
    (void)use_cpu;  // 将来の DirectML 対応のために残す

#ifdef _WIN32
    // UTF-8 -> UTF-16 (Windows) 変換
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, NULL, 0);
    std::wstring wpath(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wpath[0], size_needed);
    // wpath はヌル終端を含んでいるため、末尾の \0 を削除（もしあれば）
    if (!wpath.empty() && wpath.back() == L'\0') wpath.pop_back();
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
}

std::vector<int64_t> OnnxPolicy::infer(const uint8_t* obs_data, int64_t num_envs) {
    // 入力テンソルを構築: [num_envs, 2, 5, 6, 13] (uint8)
    std::array<int64_t, 5> input_shape{num_envs, kPlayers, kColors, kWidth, kHeight};
    const int64_t total_elems = num_envs * kObsPerEnv;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // uint8 データをそのままテンソルとして使用（定数を明示的に指定して float 誤認識を防ぐ）
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        memory_info,
        const_cast<uint8_t*>(obs_data), total_elems * sizeof(uint8_t),
        input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    );

    // 推論実行
    const char* input_names[]  = { input_name_.c_str()  };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1
    );

    // logits [num_envs, 22] → argmax → 行動インデックス
    const float* logits = output_tensors[0].GetTensorData<float>();
    auto logits_shape   = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int64_t num_actions = logits_shape[1]; // 22

    std::vector<int64_t> actions(num_envs);
    for (int64_t i = 0; i < num_envs; ++i) {
        const float* row = logits + i * num_actions;
        actions[i] = static_cast<int64_t>(
            std::max_element(row, row + num_actions) - row
        );
    }
    return actions;
}

} // namespace puyotan
