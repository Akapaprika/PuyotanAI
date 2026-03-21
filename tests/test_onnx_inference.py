import sys
from pathlib import Path
import numpy as np
import time

BASE_DIR = Path(__file__).resolve().parent.parent
for d in [BASE_DIR / "native" / "dist", BASE_DIR / "native" / "build_Release" / "Release"]:
    if d.exists():
        sys.path.insert(0, str(d))
        break

import puyotan_native as p

def main():
    model_path = str(BASE_DIR / "models" / "puyotan_final.onnx")
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    
    # 【回避策】日本語パス問題を避けるため、一時フォルダにコピー
    import shutil
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 元のファイル名を維持しないと .data が見つからない
        tmp_model_path = Path(tmpdir) / Path(model_path).name
        shutil.copy2(model_path, tmp_model_path)
        
        # .data ファイルがあればそれもコピー
        data_path = Path(model_path).with_suffix(".onnx.data")
        if data_path.exists():
            shutil.copy2(data_path, tmp_model_path.with_suffix(".onnx.data"))
            
        print(f"Copied to temp path: {tmp_model_path}")
        
        policy = p.OnnxPolicy(str(tmp_model_path), use_cpu=True)
        
        if policy.is_loaded():
            print("Model loaded successfully!")
        else:
            print("Failed to load model.")
            return

        # ダミー観測データ (N=1024)
        N = 1024
        print(f"Generating dummy observations for N={N}...")
        obs = np.zeros((N, 2, 5, 6, 13), dtype=np.uint8)
        # ランダムな値を少し入れてみる
        obs[:, :, :, 0, :5] = 1

        print("Running inference...")
        start = time.perf_counter()
        actions = policy.infer(obs)
        end = time.perf_counter()

    elapsed = end - start
    fps = N / elapsed
    print(f"Inference complete: {len(actions)} actions returned.")
    print(f"Time: {elapsed:.4f} sec  |  Throughput: {fps:.0f} obs/sec")
    print("Sample actions:", actions[:10])

if __name__ == "__main__":
    main()
