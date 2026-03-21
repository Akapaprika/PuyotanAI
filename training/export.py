"""
PyTorch モデル → ONNX 書き出しモジュール（単一責務）

使用方法:
  python -m training.export --checkpoint models/puyotan_v1.pt --output models/puyotan_v1.onnx
"""
import argparse
from pathlib import Path
import torch
from training.model import PuyotanPolicy, INPUT_DIM


def export_to_onnx(checkpoint_path: str, output_path: str):
    """
    指定した PyTorch チェックポイントを ONNX 形式で書き出す。

    Args:
        checkpoint_path: 学習済みモデルの .pt ファイルパス
        output_path: 出力先の .onnx ファイルパス
    """
    model = PuyotanPolicy()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # ダミー入力（C++ 側の observation 形状と一致させる）
    dummy_input = torch.zeros(1, 2, 5, 6, 13, dtype=torch.uint8)

    print(f"Exporting: {checkpoint_path} -> {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["logits", "value"],
        dynamic_axes={"obs": {0: "batch_size"}},  # バッチサイズを動的に
        opset_version=17,
    )
    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
