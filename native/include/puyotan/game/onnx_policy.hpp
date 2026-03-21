#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace puyotan {

/**
 * ONNX Runtime を使用した AI 推論クラス（単一責務）
 *
 * 使い方:
 *   OnnxPolicy policy("models/puyotan.onnx");
 *   auto actions = policy.infer(obs_uint8, num_envs);
 */
class OnnxPolicy {
public:
    /**
     * @param model_path .onnx ファイルへのパス
     * @param use_cpu    true: CPU推論, false: DirectML (GPU)推論
     */
    explicit OnnxPolicy(const std::string& model_path, bool use_cpu = true);

    /**
     * バッチ推論: uint8 の観測データから行動インデックスを返す
     *
     * @param obs_data  観測データ (num_envs * 2 * 5 * 6 * 13 bytes, uint8)
     * @param num_envs  並列環境数
     * @return          各環境の行動インデックス [0, 21]
     */
    std::vector<int64_t> infer(const uint8_t* obs_data, int64_t num_envs);

    /** モデルが有効に読み込まれているか */
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
