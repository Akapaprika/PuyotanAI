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

Notes
-----
- The native executables are copied into `native/dist` by the CMake build.
- The Python wrappers locate and invoke those executables; Python/C++ crossing overhead is excluded from the measured timing.
- If the executables are missing, rebuild the native binaries with `native/build.bat`.
