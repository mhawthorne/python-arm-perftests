## python-arm-perftests

Small, repeatable **performance tests to compare arm64 vs x86_64** (with NumPy-heavy workloads).

### Setup

This repo uses [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

### Running benchmarks

#### Recommended: run both architectures on Apple Silicon

On Apple Silicon with both arm64 and x86_64 Pythons installed, use `bin/run_both_arch.sh` to run the same benchmark suites under both architectures:

```bash
chmod +x bin/run_both_arch.sh
bin/run_both_arch.sh
```

The script creates separate virtual environments for each architecture, runs the benchmark suites, and generates comparison plots. It's configured via environment variables:

- **`ARM_PY`**: arm64 `python3` path (default: `/opt/homebrew/bin/python3`)
- **`X86_PY`**: x86_64 `python3` path (default: `/usr/local/bin/python3`)
- **`ARM_VENV`**: venv dir for arm64 (default: `.venv-arm64`)
- **`X86_VENV`**: venv dir for x86_64 (default: `.venv-x86_64`)
- **`OUT_DIR`**: output directory (default: `results/<timestamp>`)
- **`SUITES`**: comma-separated suites to run (default: `python,numpy`)
- **`PYTHON_ARGS`**: extra args passed to `benchmarks/run.py` (e.g. `--fast`)
- **`PLOT`**: set to `0` to skip plots (default: `1`)

Examples:

```bash
# Quick smoke runs
PYTHON_ARGS="--fast" bin/run_both_arch.sh

# Only run the pure-Python suite, skip plotting
SUITES=python PLOT=0 PYTHON_ARGS="--fast" bin/run_both_arch.sh

# Run all suites including XGBoost
SUITES=python,numpy,xgboost_train,xgboost_infer bin/run_both_arch.sh
```

#### Run benchmarks individually

You can also run benchmarks on a single architecture:

```bash
mkdir -p results

# Run all suites
uv run python benchmarks/run.py -o results/arm64.json

# Run specific suites
uv run python benchmarks/run.py --suite python -o results/python_arm64.json
uv run python benchmarks/run.py --suite numpy -o results/numpy_arm64.json
uv run python benchmarks/run.py --suite xgboost_train -o results/xgboost_train_arm64.json
uv run python benchmarks/run.py --suite xgboost_infer -o results/xgboost_infer_arm64.json
```

### XGBoost suite prerequisite (macOS)

The `xgboost_*` suites require an OpenMP runtime. On macOS, install `libomp` via Homebrew:

- **arm64 Homebrew** (Apple Silicon): `brew install libomp` (typically installs to `/opt/homebrew`)
- **x86_64 Homebrew** (Rosetta): you may also need `libomp` in the x86 Homebrew prefix (often `/usr/local`) if you run the x86_64 Python under Rosetta.

### Plot / compare results

Compare two result files:

```bash
uv run python benchmarks/plot.py results/arm64.json results/x86_64.json --out results/compare.png
```

This generates a bar chart and prints a summary (mean times + speedups).

### Tips for cleaner comparisons

- Keep **NumPy thread settings** consistent. Common env vars: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`.
- Close other heavy apps; on laptops, plug in power and disable "low power" modes if possible.

