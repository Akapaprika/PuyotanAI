#!/usr/bin/env python3
"""Statistical benchmarking orchestrator for PuyotanAI.

Runs engine and beam search benchmarks multiple times to collect samples,
calculates basic statistics, and performs Welch's t-test with 10% Trimmed Mean
to eliminate outlier noise and maximize statistical power.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
DIST_DIR = BASE_DIR / "native" / "dist"
BUILD_RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
BUILD_DEBUG_DIR = BASE_DIR / "native" / "build_Debug" / "Debug"

ENGINE_EXE = "engine_benchmark.exe" if os.name == "nt" else "engine_benchmark"
BEAM_EXE = "beam_search_benchmark.exe" if os.name == "nt" else "beam_search_benchmark"


def print_prominent_warning(title, cmd_str="", stderr_str="", stdout_str=""):
    """エラー・警告内容を詳細かつ目立つように出力する関数"""
    border = "═" * 74
    print(f"\n\033[93m╔{border}╗", file=sys.stderr)
    print(f"║ ⚠️  【連鎖点検警告】{title.ljust(50)} ║", file=sys.stderr)
    if cmd_str:
        print(f"║    実行コマンド: {cmd_str[:58].ljust(58)} ║", file=sys.stderr)
    print(f"╚{border}╝\033[0m", file=sys.stderr)
    
    if stderr_str:
        print(f"\033[91m[エラー出力 (stderr)]:\n{stderr_str.strip()}\033[0m", file=sys.stderr)
    if stdout_str:
        print(f"\033[90m[プロセス標準出力 (stdout)]:\n{stdout_str.strip()}\033[0m", file=sys.stderr)
    print("", file=sys.stderr)


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
        print_prominent_warning(
            f"プロセス異常終了 (Exit Code: {result.returncode})",
            cmd_str=" ".join(args_list),
            stderr_str=result.stderr.strip() or "（stderr は空です。アサーション失敗または異常終了）",
            stdout_str=result.stdout.strip()
        )
    return result.stdout, result.stderr, result.returncode


def parse_engine_output(stdout, stderr, returncode):
    metrics = {"engine_status": "ok" if returncode == 0 else f"error_exit_{returncode}"}
    if returncode != 0:
        metrics["engine_error"] = stderr.strip() or "unknown_error"

    if stdout:
        fps_match = re.search(r"FPS \(frames/s\):\s*([\d\.]+)", stdout)
        games_match = re.search(r"Games/s:\s*([\d\.]+)", stdout)
        
        if fps_match:
            metrics["engine_fps"] = float(fps_match.group(1))
        if games_match:
            metrics["engine_games_per_sec"] = float(games_match.group(1))
    return metrics


def parse_beam_output(stdout, stderr, returncode, prefix="beam"):
    metrics = {f"{prefix}_status": "ok" if returncode == 0 else f"error_exit_{returncode}"}
    if returncode != 0:
        metrics[f"{prefix}_error"] = stderr.strip() or f"exit_code_{returncode}"

    if stdout:
        fps_match = re.search(r"FPS \(frames/s\):\s*([\d\.]+)", stdout)
        searches_match = re.search(r"Searches/sec:\s*([\d\.]+)", stdout)
        nodes_match = re.search(r"Nodes/sec:\s*([\d\.]+)", stdout)
        
        avg_match = re.search(r"Avg:\s*([\d\.]+)\s*ms", stdout)
        p50_match = re.search(r"P50:\s*([\d\.]+)\s*ms", stdout)
        p95_match = re.search(r"P95:\s*([\d\.]+)\s*ms", stdout)
        p99_match = re.search(r"P99:\s*([\d\.]+)\s*ms", stdout)

        if fps_match:
            metrics[f"{prefix}_fps"] = float(fps_match.group(1))
        if searches_match:
            metrics[f"{prefix}_searches_per_sec"] = float(searches_match.group(1))
        if nodes_match:
            metrics[f"{prefix}_nodes_per_sec"] = float(nodes_match.group(1))
        if avg_match:
            metrics[f"{prefix}_latency_avg_ms"] = float(avg_match.group(1))
        if p50_match:
            metrics[f"{prefix}_latency_p50_ms"] = float(p50_match.group(1))
        if p95_match:
            metrics[f"{prefix}_latency_p95_ms"] = float(p95_match.group(1))
        if p99_match:
            metrics[f"{prefix}_latency_p99_ms"] = float(p99_match.group(1))
            
    return metrics


def collect_data(iter_engine, iter_light, iter_heavy, duration_engine, duration_light, duration_heavy, config_path=None):
    engine_path = find_executable(ENGINE_EXE)
    beam_path = find_executable(BEAM_EXE)
    
    if config_path is None:
        cand_cfg = BASE_DIR / "native" / "resources" / "beam_config.json"
        if cand_cfg.exists():
            config_path = str(cand_cfg.resolve())
    else:
        config_path = str(Path(config_path).resolve())
            
    print("Performing warmup run...")
    run_benchmark_once(engine_path, ["--duration", "1.0"])
    run_benchmark_once(beam_path, ["--duration", "1.0", "--beam-width", "500", "--look-ahead", "3", "--dbs", "0"])
    if config_path and duration_heavy > 0:
        run_benchmark_once(beam_path, ["--duration", "2.0", "--config", config_path])

    print(f"Starting runs (Engine: {iter_engine} runs × {duration_engine}s, Light: {iter_light} runs × {duration_light}s, Heavy: {iter_heavy} runs × {duration_heavy}s)...")
    print("Order: [Engine × N] → [Light × N] → [Heavy × N] (block-based)")

    # --- Block 1: Engine ---
    engine_results = []
    if duration_engine > 0 and iter_engine > 0:
        print(f"\n=== Engine Block ({iter_engine} runs) ===")
        for i in range(iter_engine):
            print(f"  Engine {i + 1}/{iter_engine}...")
            stdout, stderr, code = run_benchmark_once(engine_path, ["--duration", str(duration_engine)])
            engine_results.append(parse_engine_output(stdout, stderr, code))
            time.sleep(3.0)

    # --- Block 2: Beam Light ---
    light_results = []
    if duration_light > 0 and iter_light > 0:
        print(f"\n=== Beam Light Block ({iter_light} runs) ===")
        beam_light_args = ["--duration", str(duration_light), "--beam-width", "500", "--look-ahead", "10", "--dbs", "0"]
        for i in range(iter_light):
            print(f"  Light {i + 1}/{iter_light}...")
            stdout, stderr, code = run_benchmark_once(beam_path, beam_light_args)
            light_results.append(parse_beam_output(stdout, stderr, code, prefix="beam_light"))
            time.sleep(3.0)

    # --- Block 3: Beam Heavy ---
    heavy_results = []
    if duration_heavy > 0 and iter_heavy > 0 and config_path:
        print(f"\n=== Beam Heavy Block ({iter_heavy} runs) ===")
        beam_heavy_args = ["--duration", str(duration_heavy), "--config", config_path]
        for i in range(iter_heavy):
            print(f"  Heavy {i + 1}/{iter_heavy}...")
            stdout, stderr, code = run_benchmark_once(beam_path, beam_heavy_args)
            heavy_results.append(parse_beam_output(stdout, stderr, code, prefix="beam_heavy"))
            if i < iter_heavy - 1:
                time.sleep(3.0)

    # --- Merge per-iteration results ---
    max_iters = max(len(engine_results), len(light_results), len(heavy_results), 1)
    results = []
    for i in range(max_iters):
        combined = {}
        if i < len(engine_results):
            combined.update(engine_results[i])
        if i < len(light_results):
            combined.update(light_results[i])
        if i < len(heavy_results):
            combined.update(heavy_results[i])
        results.append(combined)

    return results


def trim_raw_samples(samples, trim_pct=0.1):
    n = len(samples)
    if n < 5:
        return samples
    sorted_samples = sorted(samples)
    k = int(n * trim_pct)
    if k == 0:
        k = 1
    return sorted_samples[k : n - k]


def perform_statistical_test(base_results, pr_results):
    if not base_results or not pr_results:
        return {}
    
    # 数値メトリクスのみを対象にする（_status や _error などの文字列を除外）
    keys = set()
    for r in base_results + pr_results:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                keys.add(k)
                
    comparison = {}
    import math

    trim_pct = 0.1
    
    for key in sorted(keys):
        base_raw = [r[key] for r in base_results if key in r and isinstance(r[key], (int, float))]
        pr_raw = [r[key] for r in pr_results if key in r and isinstance(r[key], (int, float))]
        
        base_samples = trim_raw_samples(base_raw, trim_pct)
        pr_samples = trim_raw_samples(pr_raw, trim_pct)
        
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
            
            df = (var_base / n_base + var_pr / n_pr) ** 2 / (
                (var_base / n_base) ** 2 / (n_base - 1) + 
                (var_pr / n_pr) ** 2 / (n_pr - 1)
            )
            
            f_val = 1.0 - 2.0 / (9.0 * df)
            denom = math.sqrt(f_val + (2.0 / (9.0 * df)) * (t_stat ** 2))
            if denom == 0:
                z_val = 0.0
            else:
                z_val = (f_val * t_stat) / denom
                
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


def get_git_info():
    """Returns current git commit hash and branch name if available."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        branch = "unknown"
    return sha, branch


def format_stat_value(val, std):
    if abs(val) < 10.0 and abs(val) > 0.0:
        return f"{val:.4f} (±{std:.4f})"
    else:
        return f"{val:.2f} (±{std:.2f})"


def generate_markdown_report(comparison, iterations, base_meta=None, pr_meta=None):
    md = []
    md.append("# PuyotanAI Performance Benchmark Report")
    md.append(f"Statistically compared using Welch's t-test over **{iterations} repetitions** of runs (with 10% outlier trimming).")
    md.append("")
    
    if base_meta or pr_meta:
        b_str = f"Commit: `{base_meta.get('commit_sha', 'unknown')}` | Branch: `{base_meta.get('branch', 'unknown')}` | Time: `{base_meta.get('timestamp', 'unknown')}`" if base_meta else "N/A"
        p_str = f"Commit: `{pr_meta.get('commit_sha', 'unknown')}` | Branch: `{pr_meta.get('branch', 'unknown')}` | Time: `{pr_meta.get('timestamp', 'unknown')}`" if pr_meta else "N/A"
        md.append(f"- **Base (比較基準)**: {b_str}")
        md.append(f"- **PR   (測定対象)**: {p_str}")
        md.append("")
    
    if not comparison:
        md.append("※ 有効な数値メトリクスが取得できませんでした（ベンチマーク異常終了のログを確認してください）。")
        return "\n".join(md)

    md.append("| Metric | Base Mean | PR Mean | Change (%) | p-value | Significant? | Status |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for key, data in comparison.items():
        metric_name = key.replace("_", " ").title()
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
        base_str = format_stat_value(data['base_mean'], data['base_std'])
        pr_str = format_stat_value(data['pr_mean'], data['pr_std'])
        
        md.append(f"| {metric_name} | {base_str} | {pr_str} | {change_str} | {p_val_str} | {sig_str} | {status} |")
        
    md.append("")
    md.append("> *Note: Significance threshold is set at $p < 0.05$. For latency metrics, lower values are better. For throughput metrics (FPS, Searches/sec, Nodes/sec), higher values are better.*")
    return "\n".join(md)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"bench_results_{timestamp}.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the benchmarks and save output")
    parser.add_argument("--compare", "-c", nargs="*", metavar="FILE", help="Compare benchmark results.")
    parser.add_argument("--iterations", type=int, default=None, help="Global repetition override for all blocks")
    parser.add_argument("--iterations-engine", type=int, default=5, help="Number of repetitions for engine benchmark (default: 5)")
    parser.add_argument("--iterations-light", type=int, default=5, help="Number of repetitions for light beam search (default: 5)")
    parser.add_argument("--iterations-heavy", type=int, default=3, help="Number of repetitions for heavy beam search (default: 3)")
    parser.add_argument("--duration-engine", type=float, default=5.0, help="Duration of engine benchmark in seconds (default: 5.0)")
    parser.add_argument("--duration-light", type=float, default=10.0, help="Duration of light beam search in seconds (default: 10.0)")
    parser.add_argument("--duration-heavy", type=float, default=240.0, help="Duration of heavy solo beam search in seconds (default: 240.0)")
    parser.add_argument("--duration", type=float, default=None, help="Legacy duration option (sets all durations to this value)")
    parser.add_argument("--config", type=str, default="native/resources/beam_config.json", help="Path to beam_config.json for heavy search (default: native/resources/beam_config.json)")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output JSON")
    parser.add_argument("--output-md", type=str, default=None, help="Path to output Markdown report")
    args = parser.parse_args()

    if args.run:
        d_engine = args.duration if args.duration is not None else args.duration_engine
        d_light = args.duration if args.duration is not None else args.duration_light
        d_heavy = args.duration if args.duration is not None else args.duration_heavy

        i_engine = args.iterations if args.iterations is not None else args.iterations_engine
        i_light = args.iterations if args.iterations is not None else args.iterations_light
        i_heavy = args.iterations if args.iterations is not None else args.iterations_heavy
        
        sha, branch = get_git_info()
        results = collect_data(i_engine, i_light, i_heavy, d_engine, d_light, d_heavy, args.config)
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "commit_sha": sha,
                "branch": branch,
                "iterations_engine": i_engine,
                "iterations_light": i_light,
                "iterations_heavy": i_heavy,
                "duration_engine": d_engine,
                "duration_light": d_light,
                "duration_heavy": d_heavy,
                "results": results
            }, f, indent=2)
        print(f"\nResults successfully saved to {args.output} (Commit: {sha}, Branch: {branch})")
        
    elif args.compare is not None:
        num_args = len(args.compare)
        bench_files = sorted([str(p) for p in Path(".").glob("bench_results_*.json")])
        
        if num_args == 0:
            if len(bench_files) < 2:
                print("Error: Auto-compare requires at least 2 saved benchmark files in the current directory.", file=sys.stderr)
                sys.exit(1)
            base_path = bench_files[-2]
            pr_path = bench_files[-1]
            print(f"Auto-comparing two newest results:\n  Base (Previous): {base_path}\n  PR   (Newest)  : {pr_path}\n")
        elif num_args == 1:
            if len(bench_files) < 1:
                print("Error: No local benchmark files found to compare against.", file=sys.stderr)
                sys.exit(1)
            base_path = args.compare[0]
            pr_path = bench_files[-1]
            print(f"Comparing specified base against newest result:\n  Base: {base_path}\n  PR   (Newest): {pr_path}\n")
        elif num_args == 2:
            base_path = args.compare[0]
            pr_path = args.compare[1]
        else:
            print("Error: --compare / -c takes at most 2 arguments (base, pr).", file=sys.stderr)
            sys.exit(1)

        with open(base_path) as f:
            base_data = json.load(f)
        with open(pr_path) as f:
            pr_data = json.load(f)
            
        base_meta = {
            "commit_sha": base_data.get("commit_sha", "unknown"),
            "branch": base_data.get("branch", "unknown"),
            "timestamp": base_data.get("timestamp", "unknown"),
            "file": base_path
        }
        pr_meta = {
            "commit_sha": pr_data.get("commit_sha", "unknown"),
            "branch": pr_data.get("branch", "unknown"),
            "timestamp": pr_data.get("timestamp", "unknown"),
            "file": pr_path
        }

        iterations = base_data.get("iterations", 20)
        comparison = perform_statistical_test(base_data["results"], pr_data["results"])
        report = generate_markdown_report(comparison, iterations, base_meta, pr_meta)
        
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