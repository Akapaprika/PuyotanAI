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

    import warnings
    import logging
    import os
    # Suppress all the noisy PyTorch/ONNX internals
    warnings.filterwarnings("ignore")
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    
    # Force standard export to avoid Dynamo's Unicode print issues on Windows
    model.eval()
    dummy_input = torch.zeros(1, 2, 5, 6, 14, dtype=torch.uint8)

    print(f"Exporting: {output_path}")
    
    # Standard export is safer on Windows CP932 environments
    # We use explicit suppression to keep the console clean.
    # torch.onnx.export は大きいモデルで外部データ(.data)を一時パスへ書き出す場合がある。
    # 一旦 tmp ファイルへ書き出してから onnx でインライン化して最終パスへ保存する。
    import tempfile, os
    import onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_onnx = os.path.join(tmpdir, "tmp_export.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            tmp_onnx,
            input_names=["obs"],
            output_names=["logits", "value"],
            dynamic_axes={"obs": {0: "batch_size"}},
            opset_version=18,
            verbose=False,
        )

        # 外部データがある場合でも全部インラインにまとめて読み込む
        onnx_model = onnx.load(tmp_onnx, load_external_data=True)

    # 全テンソルをインライン化して単体ファイルとして保存（.data なし）
    onnx.save(onnx_model, output_path,
              save_as_external_data=False)

    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
