Performance tests for PuyotanAI

Usage
-----

These scripts now invoke native C++ benchmark executables so timing reflects only C++ work.

Build the native targets first:

```powershell
cd native
.\build.bat
```

Run the engine benchmark:

```powershell
python -m tests.perf.bench_engine --duration 10
```

Run the beam-search benchmark:

```powershell
python -m tests.perf.bench_beam --duration 10
```

Run the regression checks in the native benchmarks:

```powershell
python -m tests.perf.bench_engine --regression
python -m tests.perf.bench_beam --regression
```

Run the statistical benchmark & comparison:

```powershell
# Run PR branch benchmarks (e.g. 30 runs of 10 seconds each)
python -m tests.perf.run_stat_bench --run --iterations 30 --duration 10 --output pr_results.json

# Run Base branch benchmarks
python -m tests.perf.run_stat_bench --run --iterations 30 --duration 10 --output base_results.json

# Compare them and perform Welch's t-test
python -m tests.perf.run_stat_bench --compare base_results.json pr_results.json --output-md comparison.md
```

Notes
-----
- The native executables are copied into `native/dist` by the CMake build.
- The Python wrappers locate and invoke those executables; Python/C++ crossing overhead is excluded from the measured timing.
- If the executables are missing, rebuild the native binaries with `native/build.bat`.
