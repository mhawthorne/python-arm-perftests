# AGENTS.md

## Purpose

This repository contains **repeatable micro-/meso-benchmarks** intended to compare **Python + NumPy performance across architectures**, primarily **arm64 vs x86_64**.

The goal is to produce:
- **Comparable, machine-readable results** (`pyperf` JSON) that can be collected on different machines.
- **Simple visualization** (bar chart) for quick inspection of relative performance.
- **Useful metadata** (CPU/arch + NumPy build hints + thread env vars) to make results interpretable.

### Primary workflow goal

Run the **same benchmark suites** under **arm64** and **x86_64** Python runtimes, producing two result JSON files that can be compared/visualized.

On Apple Silicon, “x86_64” typically means running an x86_64 Python under **Rosetta 2**. On a real x86_64 machine, it’s native.

## What’s in here

- `pyproject.toml`, `uv.lock`: `uv`-managed project + locked dependencies.
- `benchmarks/run.py`: Runs a benchmark suite using `pyperf` and writes results with `-o/--output`.
- `benchmarks/plot.py`: Loads two `pyperf` JSON files and creates a comparison plot.
- `results/`: Intended output directory for benchmark JSON and plots (kept via `.gitkeep`).

## How to run (typical workflow)

Run on each machine/architecture you want to compare:

```bash
cd /Users/m3h/yes/hacks-local/python-arm-perftests
uv sync
mkdir -p results

# On an arm64 Python
uv run -p /path/to/arm64/python3 python benchmarks/run.py -o results/arm64.json

# On an x86_64 Python (Rosetta 2 on Apple Silicon, or native on x86_64 hardware)
uv run -p /path/to/x86_64/python3 python benchmarks/run.py -o results/x86_64.json

# Compare/plot
uv run python benchmarks/plot.py results/arm64.json results/x86_64.json --out results/compare.png
```

### Convenience: run both architectures on one machine (two venvs)

If you have both an arm64 Python and an x86_64 Python installed (e.g. Apple Silicon + Rosetta),
use `bin/

both_arch.sh` to create **two separate venvs** and run the same suites under both:

```bash
chmod +x bin/run_both_arch.sh
bin/run_both_arch.sh
```

`bin/run_both_arch.sh` is configured via environment variables:

- **`ARM_PY`**: arm64 `python3` path (default: `/opt/homebrew/bin/python3`)
- **`X86_PY`**: x86_64 `python3` path (default: `/usr/local/bin/python3`)
- **`ARM_VENV`**: venv dir for arm64 (default: `.venv-arm64`)
- **`X86_VENV`**: venv dir for x86_64 (default: `.venv-x86_64`)
- **`OUT_DIR`**: results directory (default: `results`)
- **`SUITES`**: comma-separated suites to run (default: `python,numpy`)
- **`PYTHON_ARGS`**: extra args passed through to `benchmarks/run.py` (e.g. `--fast` or `--rigorous`)
- **`PLOT`**: set to `0` to skip plots (default: `1`)

Examples:

```bash
# Quick smoke runs
PYTHON_ARGS="--fast" bin/run_both_arch.sh

# Only run the pure-Python suite, skip plotting
SUITES=python PLOT=0 PYTHON_ARGS="--fast" bin/run_both_arch.sh
```

Sanity-check which architecture you actually ran:

```bash
uv run -p /path/to/python3 python -c "import platform,sys; print(platform.machine()); print(sys.executable)"
```

You can run specific suites (e.g. pure-Python microbenchmarks vs NumPy-heavy):

```bash
uv run python benchmarks/run.py --suite python -o results/python_arm64.json
uv run python benchmarks/run.py --suite numpy -o results/numpy_arm64.json
uv run python benchmarks/run.py --suite xgboost_train -o results/xgboost_train_arm64.json
uv run python benchmarks/run.py --suite xgboost_infer -o results/xgboost_infer_arm64.json
```

### XGBoost suite prerequisite (macOS)

The `xgboost_*` suites require an OpenMP runtime. On macOS, install `libomp` via Homebrew.

On Apple Silicon running **both** arm64 and x86_64 Pythons, you may need `libomp` available in **both** Homebrew prefixes:
- arm64 Homebrew typically installs to `/opt/homebrew`
- x86_64 (Rosetta) Homebrew typically installs to `/usr/local`

## Reproducibility guidelines (important for fair comparisons)

When adding or running benchmarks, try to keep these consistent across runs:
- **Python version** (and build) and **NumPy version**.
- **NumPy/BLAS backend** (Accelerate/OpenBLAS/MKL) — this can dominate performance.
- **Threading env vars** (common ones: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`).
- **Power/thermal state** (plug in laptops, avoid low-power mode, minimize background load).

Prefer **fewer moving parts** over “more realistic” scenarios when the question is architecture deltas.

### Recommended run settings (stability vs speed)

The right settings depend on how noisy your machine is (thermals, background load).

- **Quick smoke test**: run with `--fast` just to validate everything works end-to-end.
- **Typical comparison**: prefer longer runs (or more loops) so each benchmark has enough signal above timer noise.
  - If you’re using `pyperf` defaults, you’ll usually get reasonable statistics, but small/fast ops can still be noisy.
  - If you change durations/loops, keep them identical across the two architectures.

### Thermal throttling / power management gotchas

To avoid misleading “architecture deltas” that are really power/thermal differences:

- **Keep conditions consistent**: same power source, similar ambient temperature, same laptop mode (avoid low-power mode).
- **Minimize background load**: close heavy apps, disable indexing/backup jobs if possible.
- **Detect drift**: if you see later benchmarks consistently slower than earlier ones within the same run, you may be throttling.
  - Re-run after a cool-down, or run suites separately and compare within-suite stability.

### How to interpret results (quick mental model)

- Benchmarks are compared by **mean time** (lower is better).
- When plotting arm64 vs x86_64:
  - A ratio \(b/a > 1\) means **b is slower** than a for that benchmark (assuming b is the numerator in the printed ratio).
  - Large wins/losses in NumPy-heavy tests often indicate **BLAS backend / threading** differences more than pure ISA differences.

## Adding benchmarks

Benchmarks live in `benchmarks/run.py`. Keep these rules:
- **Deterministic inputs**: seed RNGs and reuse allocated arrays where possible.
- **Avoid measuring allocations you don’t care about**: pre-allocate output arrays for ops that support `out=`.
- **Pick stable sizes**: large enough to avoid timer noise, not so large that memory pressure dominates.
- **Name benchmarks clearly**: include operation + size + dtype (e.g. `numpy.matmul[512x512,f32]`).
- **Prefer `pyperf.Runner.bench_func(...)`** for pure-Python callables; use `inner_loops` to reduce overhead.

If you introduce a new suite, wire it behind a `--suite` option (so results are easy to segment).

## Result format and comparison

- `benchmarks/run.py` writes **`pyperf` JSON** (via `--output`).
- `benchmarks/plot.py` computes **mean time (seconds)** per benchmark and plots two series.
- Lower times are better. The script also prints a simple `b/a` ratio per benchmark.

If you need more statistical detail, use `pyperf`’s tooling on a result file:

```bash
uv run python -m pyperf stats results/arm64.json
uv run python -m pyperf hist results/arm64.json
```


