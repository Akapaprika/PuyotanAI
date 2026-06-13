#!/usr/bin/env python3
"""Engine-only performance benchmark wrapper.

This script invokes the native C++ benchmark executable so timing reflects only C++ work.
"""
import argparse
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DIST_DIR = BASE_DIR / "native" / "dist"
BUILD_RELEASE_DIR = BASE_DIR / "native" / "build_Release" / "Release"
BUILD_DEBUG_DIR = BASE_DIR / "native" / "build_Debug" / "Debug"
EXECUTABLE = "engine_benchmark.exe" if os.name == "nt" else "engine_benchmark"


def get_executable_path():
    candidates = [DIST_DIR / EXECUTABLE, BUILD_RELEASE_DIR / EXECUTABLE, BUILD_DEBUG_DIR / EXECUTABLE]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Native benchmark executable not found in any expected location:\n"
        f"  {candidates[0]}\n"
        f"  {candidates[1]}\n"
        f"  {candidates[2]}\n"
        "Build the native benchmark target with native/build.bat and ensure the executable is generated."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds")
    parser.add_argument("--regression", action="store_true", help="Run the native regression test")
    args = parser.parse_args()

    exe_path = get_executable_path()
    cmd = [str(exe_path), "--duration", str(args.duration)]

    if args.regression:
        cmd.append("--regression")

    env = os.environ.copy()
    env["PATH"] = str(DIST_DIR) + os.pathsep + env.get("PATH", "")

    completed = subprocess.run(cmd, env=env, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
