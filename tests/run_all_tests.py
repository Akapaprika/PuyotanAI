"""
tests/run_all_tests.py

PuyotanAI プロジェクト全体の個別のテストモジュールを一括統括・実行するマスターテストランナー。
"""
import sys
import os
import time
import subprocess
from pathlib import Path

# パス設定
_PROJECT_ROOT = Path(__file__).parent.parent
_BENCHMARK_EXE = _PROJECT_ROOT / "native" / "build_Release" / "Release" / "beam_search_benchmark.exe"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# カラー表示ユーティリティ
def green(text): return f"\033[92m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0

    def run_module(self, name: str, main_func):
        self.total += 1
        print(f"[{self.total:02d}] {name:<45}", end="", flush=True)
        t0 = time.time()
        try:
            main_func()
            elapsed = time.time() - t0
            print(f" ... [{green('PASS')}] ({elapsed:.2f}s)")
            self.passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ... [{red('FAIL')}] ({elapsed:.2f}s)")
            print(f"     {yellow('Error:')} {e}")
            self.failed += 1

    def run_executable(self, name: str, cmd: list):
        self.total += 1
        print(f"[{self.total:02d}] {name:<45}", end="", flush=True)
        t0 = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            assert res.returncode == 0, f"Return code {res.returncode}:\n{res.stdout}\n{res.stderr}"
            assert "Regression test PASSED!" in res.stdout
            elapsed = time.time() - t0
            print(f" ... [{green('PASS')}] ({elapsed:.2f}s)")
            self.passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ... [{red('FAIL')}] ({elapsed:.2f}s)")
            print(f"     {yellow('Error:')} {e}")
            self.failed += 1

def main():
    print("\n" + "=" * 70)
    print("  PuyotanAI Master Verification Suite (World Strongest AI Foundation)")
    print("=" * 70)

    runner = TestRunner()

    # 1. 個別テストモジュールのインポートと実行
    import tests.test_board_and_chain as t_board
    runner.run_module("1. Board, BitBoard & Chain Core", t_board.run_all)

    import tests.test_engine_match as t_match
    runner.run_module("2. PuyotanMatch Engine Mechanics", t_match.run_all)

    import tests.test_config_loader as t_cfg
    runner.run_module("3. BeamConfigLoader JSON Parser", t_cfg.run_all)

    import tests.test_gui_agents as t_gui
    runner.run_module("4. Unified AI Agent Strategy Classes", t_gui.run_all)

    import tests.test_agent_factory as t_factory
    runner.run_module("5. AgentFactory Mode & Instantiation", t_factory.run_all)

    import tests.test_solo_regression as t_solo
    runner.run_module("6. Solo AI Golden Master (50-steps)", t_solo.verify_against_golden)

    # 2. C++ ネイティブバイナリの回帰テスト
    if _BENCHMARK_EXE.exists():
        runner.run_executable("9. Native C++ Executable Benchmark (-r)", [str(_BENCHMARK_EXE), "-r"])

    print("=" * 70)
    if runner.failed == 0:
        print(green(f"  ALL {runner.passed}/{runner.total} TEST MODULES PASSED! Project is 100% stable."))
    else:
        print(red(f"  {runner.failed}/{runner.total} TEST MODULES FAILED! Please check above."))
    print("=" * 70 + "\n")

    sys.exit(0 if runner.failed == 0 else 1)

if __name__ == "__main__":
    main()
