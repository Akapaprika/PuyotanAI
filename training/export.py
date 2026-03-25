"""
PyTorch モデル → ONNX 書き出しモジュール（単一責務）

使用方法:
  python -m training.export --checkpoint models/puyotan_v1.pt --output models/puyotan_v1.onnx
"""
import argparse
from pathlib import Path
import torch
from training.model import PuyotanPolicy


def export_to_onnx(model_or_path, output_path: str, hidden_dim=128):
    """
    指定した PyTorch モデル（またはチェックポイントパス）を ONNX 形式で書き出す。
    """
    if isinstance(model_or_path, (str, Path)):
        model = PuyotanPolicy(hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(model_or_path, map_location="cpu"))
        print(f"Exporting: {model_or_path} -> {output_path}")
    else:
        model = model_or_path
        print(f"Exporting model object -> {output_path}")

    model.eval()
    dummy_input = torch.zeros(1, 2, 5, 6, 13, dtype=torch.uint8)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["logits", "value"],
        dynamic_axes={"obs": {0: "batch_size"}},
        opset_version=18,
        verbose=False,
    )
    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
