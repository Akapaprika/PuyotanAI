"""
PyTorch モデル → ONNX 書き出しモジュール（単一責務）

使用方法:
  # 特定のファイルを変換
  python -m training.export --checkpoint models/puyotan_solo.pt --output models/puyotan_solo.onnx
  
  # models ディレクトリをスキャンして全ての .pt を .onnx に変換
  python -m training.export
"""
import argparse
from pathlib import Path
import torch
import sys
import os

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

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


def find_and_export_all(models_dir: Path):
    """
    指定したディレクトリ配下の .pt ファイルを再帰的に探し、対応する .onnx を生成する。
    """
    print(f"Scanning directory: {models_dir}")
    pt_files = list(models_dir.rglob("*.pt"))
    if not pt_files:
        print("No .pt files found.")
        return

    for pt in pt_files:
        onnx_path = pt.with_suffix(".onnx")
        try:
            export_to_onnx(pt, str(onnx_path))
        except Exception as e:
            print(f"Failed to export {pt}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to .pt file (optional if scanning)")
    parser.add_argument("--output", help="Path to output .onnx file (optional if scanning)")
    parser.add_argument("--models_dir", default=str(BASE_DIR / "models"), help="Directory to scan for .pt files")
    args = parser.parse_args()

    if args.checkpoint:
        # 手動指定モード
        output = args.output or str(Path(args.checkpoint).with_suffix(".onnx"))
        export_to_onnx(args.checkpoint, output)
    else:
        # スキャンモード
        find_and_export_all(Path(args.models_dir))
