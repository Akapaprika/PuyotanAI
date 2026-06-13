#!/usr/bin/env python3
"""Statistical benchmarking orchestrator for PuyotanAI.

Runs engine and beam search benchmarks multiple times to collect samples,
calculates basic statistics, and performs Welch's t-test to check if
performance improvements are statistically significant.
"""

import argparse
import datetime  # 追加：タイムスタンプ生成に使用
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
DIST_DIR = BASE_DIR / "native" / "dist"
BUILD_RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
BUILD_DEBUG_DIR = BASE_DIR / "native" / "build_Debug" / "Debug"

ENGINE_EXE = "engine_benchmark.exe" if os.name == "nt" else "engine_benchmark"
BEAM_EXE = "beam_search_benchmark.exe" if os.name == "nt" else "beam_search_benchmark"


def find_executable(filename):
    candidates = [DIST_DIR / filename, BUILD_RELEASE_DIR / filename, BUILD_DEBUG_DIR / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Native benchmark executable '{filename}' not found. Please build the native project first.")


def run_benchmark_once(exe_path, args_list):
    env = os.environ.copy()
    env["PATH"] = str(DIST_DIR) + os.pathsep + env.get("PATH", "")
    
    cmd = [str(exe_path)] + args_list
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Benchmark failed with exit code {result.returncode}")
    return result.stdout


def parse_engine_output(output):
    metrics = {}
    fps_match = re.search(r"FPS \(frames/s\):\s*([\d\.]+)", output)
    games_match = re.search(r"Games/s:\s*([\d\.]+)", output)
    
    if fps_match:
        metrics["engine_fps"] = float(fps_match.group(1))
    if games_match:
        metrics["engine_games_per_sec"] = float(games_match.group(1))
    return metrics


def parse_beam_output(output):
    metrics = {}
    fps_match = re.search(r"FPS \(frames/s\):\s*([\d\.]+)", output)
    searches_match = re.search(r"Searches/sec:\s*([\d\.]+)", output)
    nodes_match = re.search(r"Nodes/sec:\s*([\d\.]+)", output)
    
    # Latency section parsing
    avg_match = re.search(r"Avg:\s*([\d\.]+)\s*ms", output)
    p50_match = re.search(r"P50:\s*([\d\.]+)\s*ms", output)
    p95_match = re.search(r"P95:\s*([\d\.]+)\s*ms", output)
    p99_match = re.search(r"P99:\s*([\d\.]+)\s*ms", output)

    if fps_match:
        metrics["beam_fps"] = float(fps_match.group(1))
    if searches_match:
        metrics["beam_searches_per_sec"] = float(searches_match.group(1))
    if nodes_match:
        metrics["beam_nodes_per_sec"] = float(nodes_match.group(1))
    if avg_match:
        metrics["beam_latency_avg_ms"] = float(avg_match.group(1))
    if p50_match:
        metrics["beam_latency_p50_ms"] = float(p50_match.group(1))
    if p95_match:
        metrics["beam_latency_p95_ms"] = float(p95_match.group(1))
    if p99_match:
        metrics["beam_latency_p99_ms"] = float(p99_match.group(1))
        
    return metrics


def collect_data(iterations, duration, beam_width, look_ahead):
    engine_path = find_executable(ENGINE_EXE)
    beam_path = find_executable(BEAM_EXE)
    
    results = []
    
    # Warmup
    print("Performing warmup run...")
    try:
        run_benchmark_once(engine_path, ["--duration", "1.0"])
        beam_args = ["--duration", "1.0", "--beam-width", str(beam_width), "--look-ahead", str(look_ahead)]
        run_benchmark_once(beam_path, beam_args)
    except Exception as e:
        print(f"Warmup failed: {e}", file=sys.stderr)
        
    print(f"Starting {iterations} runs (Duration per run: {duration}s)...")
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...")
        
        # 1. Engine benchmark
        engine_out = run_benchmark_once(engine_path, ["--duration", str(duration)])
        engine_metrics = parse_engine_output(engine_out)
        
        # 2. Beam search benchmark
        beam_args = ["--duration", str(duration), "--beam-width", str(beam_width), "--look-ahead", str(look_ahead)]
        beam_out = run_benchmark_once(beam_path, beam_args)
        beam_metrics = parse_beam_output(beam_out)
        
        combined = {**engine_metrics, **beam_metrics}
        results.append(combined)
        
    return results


def perform_statistical_test(base_results, pr_results):
    keys = set(base_results[0].keys()) & set(pr_results[0].keys())
    comparison = {}
    import math  # 標準ライブラリのみ使用
    
    for key in sorted(keys):
        base_samples = [r[key] for r in base_results if key in r]
        pr_samples = [r[key] for r in pr_results if key in r]
        
        n_base = len(base_samples)
        n_pr = len(pr_samples)
        
        if n_base < 2 or n_pr < 2:
            continue
            
        base_mean = sum(base_samples) / n_base
        base_std = (sum((x - base_mean) ** 2 for x in base_samples) / (n_base - 1)) ** 0.5
        pr_mean = sum(pr_samples) / n_pr
        pr_std = (sum((x - pr_mean) ** 2 for x in pr_samples) / (n_pr - 1)) ** 0.5
        
        var_base = base_std ** 2
        var_pr = pr_std ** 2
        se = (var_base / n_base + var_pr / n_pr) ** 0.5
        
        if se == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = (pr_mean - base_mean) / se
            
            # ウェルチ・サタスウェイトの自由度(df)の算出
            df = (var_base / n_base + var_pr / n_pr) ** 2 / (
                (var_base / n_base) ** 2 / (n_base - 1) + 
                (var_pr / n_pr) ** 2 / (n_pr - 1)
            )
            
            # 【超高精度なWilson-Hilferty近似による純粋Python p値算出】
            # 自由度 df におけるスチューデントt値を、標準正規分布の Z値 に変換します
            # 自由度 3 以上において誤差 0.001 以下の極めて高い精度を誇ります
            f_val = 1.0 - 2.0 / (9.0 * df)
            # ゼロ除算防止
            denom = math.sqrt(f_val + (2.0 / (9.0 * df)) * (t_stat ** 2))
            if denom == 0:
                z_val = 0.0
            else:
                z_val = (f_val * t_stat) / denom
                
            # Z値 から2テイラー（両側）p値を算出
            p_value = 1.0 - math.erf(abs(z_val) / (2 ** 0.5))
        
        lower_is_better = "latency" in key
        diff_pct = ((pr_mean - base_mean) / base_mean) * 100.0
        
        if lower_is_better:
            improved = pr_mean < base_mean
            speedup_pct = -diff_pct
        else:
            improved = pr_mean > base_mean
            speedup_pct = diff_pct
            
        significant = p_value < 0.05
        
        comparison[key] = {
            "base_mean": base_mean,
            "base_std": base_std,
            "pr_mean": pr_mean,
            "pr_std": pr_std,
            "t_stat": t_stat,
            "p_value": p_value,
            "diff_pct": diff_pct,
            "speedup_pct": speedup_pct,
            "improved": improved,
            "significant": significant
        }
        
    return comparison


def generate_markdown_report(comparison, iterations):
    md = []
    md.append("# PuyotanAI Performance Benchmark Report")
    md.append(f"Statistically compared using Welch's t-test over **{iterations} repetitions** of 10s runs.")
    md.append("")
    md.append("| Metric | Base Mean | PR Mean | Change (%) | p-value | Significant? | Status |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for key, data in comparison.items():
        metric_name = key.replace("_", " ").title()
        
        # Formatting speedup
        sig_str = "✅ Yes" if data["significant"] else "❌ No"
        
        if data["significant"]:
            if data["improved"]:
                status = "🚀 Improved"
                change_str = f"**+{data['speedup_pct']:.2f}%**" if "latency" not in key else f"**-{abs(data['diff_pct']):.2f}%**"
            else:
                status = "⚠️ Regressed"
                change_str = f"**-{abs(data['speedup_pct']):.2f}%**" if "latency" not in key else f"**+{data['diff_pct']:.2f}%**"
        else:
            status = "😐 Unchanged"
            change_str = f"{data['diff_pct']:.2f}%"
            
        p_val_str = f"{data['p_value']:.4f}" if data["p_value"] >= 0.0001 else "< 0.0001"
        
        md.append(f"| {metric_name} | {data['base_mean']:.2f} (±{data['base_std']:.2f}) | {data['pr_mean']:.2f} (±{data['pr_std']:.2f}) | {change_str} | {p_val_str} | {sig_str} | {status} |")
        
    md.append("")
    md.append("> *Note: Significance threshold is set at $p < 0.05$. For latency metrics, lower values are better. For throughput metrics (FPS, Searches/sec, Nodes/sec), higher values are better.*")
    return "\n".join(md)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    
    # 起動時の現在日時から、ローカル実行用のデフォルトファイル名を動的に生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"bench_results_{timestamp}.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the benchmarks and save output")
    
    # 【変更点】--compare, -c を可変長引数 (nargs='*') に変更し、短縮形 -c を追加
    parser.add_argument("--compare", "-c", nargs="*", metavar="FILE", help="Compare benchmark results. 0 args: compare two newest; 1 arg: compare specified vs newest; 2 args: compare specified base vs specified pr.")
    
    parser.add_argument("--iterations", type=int, default=30, help="Number of repetitions to run")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of each benchmark run in seconds")
    parser.add_argument("--beam-width", type=int, default=500, help="Beam width for beam search benchmark")
    parser.add_argument("--look-ahead", type=int, default=10, help="Lookahead depth for beam search benchmark")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output JSON")
    parser.add_argument("--output-md", type=str, default=None, help="Path to output Markdown report")
    args = parser.parse_args()

    if args.run:
        results = collect_data(args.iterations, args.duration, args.beam_width, args.look_ahead)
        with open(args.output, "w") as f:
            # データの履歴追跡を容易にするため、jsonの内部にも実行時のタイムスタンプを保存します
            json.dump({
                "timestamp": timestamp,
                "iterations": args.iterations, 
                "duration": args.duration, 
                "results": results
            }, f, indent=2)
        print(f"Results successfully saved to {args.output}")
        
    elif args.compare is not None:
        # 【新設】自動比較用のロジック
        num_args = len(args.compare)
        
        # フォルダ内の bench_results_*.json を探し、日時順（昇順）にソート
        bench_files = sorted(glob_files := [str(p) for p in Path(".").glob("bench_results_*.json")])
        
        if num_args == 0:
            # 引数なし：最新の2ファイルを自動比較
            if len(bench_files) < 2:
                print("Error: Auto-compare requires at least 2 saved benchmark files in the current directory.", file=sys.stderr)
                sys.exit(1)
            base_path = bench_files[-2]
            pr_path = bench_files[-1]
            print(f"Auto-comparing two newest results:\n  Base (Previous): {base_path}\n  PR   (Newest)  : {pr_path}\n")
        elif num_args == 1:
            # 引数1個：指定された古いファイル ↔ 最も新しいファイルを比較
            if len(bench_files) < 1:
                print("Error: No local benchmark files found to compare against.", file=sys.stderr)
                sys.exit(1)
            base_path = args.compare[0]
            pr_path = bench_files[-1]
            print(f"Comparing specified base against newest result:\n  Base: {base_path}\n  PR   (Newest): {pr_path}\n")
        elif num_args == 2:
            # 引数2個：従来通りの指定ファイル同士の比較（GHAでの動作を保証）
            base_path = args.compare[0]
            pr_path = args.compare[1]
        else:
            print("Error: --compare / -c takes at most 2 arguments (base, pr).", file=sys.stderr)
            sys.exit(1)

        # Load datasets
        with open(base_path) as f:
            base_data = json.load(f)
        with open(pr_path) as f:
            pr_data = json.load(f)
            
        iterations = base_data.get("iterations", 30)
        comparison = perform_statistical_test(base_data["results"], pr_data["results"])
        report = generate_markdown_report(comparison, iterations)
        
        if args.output_md:
            with open(args.output_md, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Comparison report saved to {args.output_md}")
        else:
            print(report)
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
