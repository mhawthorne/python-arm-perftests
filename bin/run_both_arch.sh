#!/usr/bin/env bash
set -euo pipefail

# Run benchmark suites under arm64 and x86_64 Pythons (Rosetta 2 on Apple Silicon),
# producing comparable pyperf JSON outputs and optional plots.
#
# Defaults are the common Homebrew paths on Apple Silicon:
# - arm64:  /opt/homebrew/bin/python3
# - x86_64: /usr/local/bin/python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Install uv first, then re-run." >&2
  echo "See: https://docs.astral.sh/uv/" >&2
  exit 127
fi

ARM_PY="${ARM_PY:-/opt/homebrew/bin/python3}"
X86_PY="${X86_PY:-/usr/local/bin/python3}"

ARM_VENV="${ARM_VENV:-.venv-arm64}"
X86_VENV="${X86_VENV:-.venv-x86_64}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results}"
SUITES="${SUITES:-python,numpy}"

PYTHON_ARGS="${PYTHON_ARGS:-}"  # e.g. "--fast" or "--rigorous"
PLOT="${PLOT:-1}"              # set to 0 to skip plots

usage() {
  cat <<EOF
Usage:
  ARM_PY=... X86_PY=... $0

Env vars (optional):
  ARM_PY        Path to arm64 python3 (default: /opt/homebrew/bin/python3)
  X86_PY        Path to x86_64 python3 (default: /usr/local/bin/python3)
  ARM_VENV      Venv directory for arm64 (default: .venv-arm64)
  X86_VENV      Venv directory for x86_64 (default: .venv-x86_64)
  OUT_DIR       Output directory for results (default: results/<timestamp>)
  BASE_RESULTS_DIR Base results directory if OUT_DIR is not set (default: results)
  SUITES        Comma-separated suites to run (default: python,numpy)
  PYTHON_ARGS   Extra args passed to benchmarks/run.py (e.g. "--fast" or "--rigorous")
  PLOT          1 to generate plots, 0 to skip (default: 1)

Examples:
  $0
  PYTHON_ARGS="--fast" $0
  SUITES=python PYTHON_ARGS="--fast" $0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${OUT_DIR:-}" ]]; then
  TS="$(date +%Y%m%d%H%M%S)"
  OUT_DIR="${BASE_RESULTS_DIR}/${TS}"
fi

mkdir -p "$OUT_DIR"
echo "==> Output dir: $OUT_DIR"

THREAD_ENV_VARS=(
  OMP_NUM_THREADS
  OPENBLAS_NUM_THREADS
  MKL_NUM_THREADS
  VECLIB_MAXIMUM_THREADS
  NUMEXPR_NUM_THREADS
)

echo "==> Capturing thread env vars"
{
  echo "# Threading env vars (captured at run time)"
  for v in "${THREAD_ENV_VARS[@]}"; do
    if [[ -n "${!v:-}" ]]; then
      echo "$v=${!v}"
    else
      echo "$v="
    fi
  done
} >"$OUT_DIR/thread_env.txt"

warn_if_thread_env_set() {
  local found=0
  for v in "${THREAD_ENV_VARS[@]}"; do
    if [[ -n "${!v:-}" ]]; then
      found=1
    fi
  done
  if [[ "$found" == "1" ]]; then
    echo "==> NOTE: Some thread env vars are set in your shell; this can dominate NumPy performance."
    echo "==>       See: $OUT_DIR/thread_env.txt"
  fi
}

warn_if_thread_env_set

echo "==> Creating venvs (if needed)"
uv venv --allow-existing -p "$ARM_PY" "$ARM_VENV" >/dev/null
uv venv --allow-existing -p "$X86_PY" "$X86_VENV" >/dev/null

ensure_arch() {
  local venv="$1"
  local expected="$2"
  local py="$3"

  if [[ ! -x "$venv/bin/python3" ]]; then
    echo "ERROR: missing $venv/bin/python3"
    exit 1
  fi

  local actual
  actual="$("$venv/bin/python3" -c "import platform; print(platform.machine())" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    echo "==> Recreating $venv (expected $expected, got ${actual:-<unknown>})"
    uv venv -c -p "$py" "$venv" >/dev/null
    actual="$("$venv/bin/python3" -c "import platform; print(platform.machine())")"
    if [[ "$actual" != "$expected" ]]; then
      echo "ERROR: venv $venv is $actual but expected $expected (check ARM_PY/X86_PY)"
      exit 1
    fi
  fi
}

ensure_arch "$ARM_VENV" "arm64" "$ARM_PY"
ensure_arch "$X86_VENV" "x86_64" "$X86_PY"

run_in_env() {
  local venv="$1"
  local label="$2"
  local py="$3"
  shift 3

  # shellcheck disable=SC1090
  source "$venv/bin/activate"
  echo "==> Sync ($label)"
  # Pin the interpreter for the environment; otherwise uv may recreate the venv
  # using a different Python than the one the venv was created with (e.g. arm64).
  uv sync --active -p "$py" >/dev/null

  # Re-check after sync in case the environment was recreated.
  if [[ "$("$venv/bin/python3" -c "import platform; print(platform.machine())")" != "$label" ]]; then
    echo "ERROR: $label env is not $label after sync (check ARM_PY/X86_PY)"
    exit 1
  fi

  echo "==> Sanity check ($label)"
  python -c "import platform,sys; print('machine=', platform.machine()); print('executable=', sys.executable); print('prefix=', sys.prefix)"

  echo "==> Capture NumPy build info ($label)"
  python -c "import json, platform, sys; print(json.dumps({'platform.machine': platform.machine(), 'sys.executable': sys.executable, 'sys.version': sys.version}, indent=2, sort_keys=True))" \
    >"$OUT_DIR/runtime_${label}.json"
  python -c "import numpy as np; np.show_config()" >"$OUT_DIR/numpy_show_config_${label}.txt" 2>&1 || true
  python -c "import numpy as np; print('numpy.__version__=', np.__version__); print('numpy.__file__=', np.__file__)" \
    >"$OUT_DIR/numpy_version_${label}.txt" 2>&1 || true

  echo "==> Run suites ($label)"
  for suite in "$@"; do
    local out="$OUT_DIR/${suite}_${label}.json"
    rm -f "$out"
    # Run directly under the venv interpreter to avoid uv recreating/switching envs
    # (which can dominate runtime and also ruin arch correctness).
    # NOTE: pyperf's canonical output flag is `-o`. (Some versions also accept
    # `--output`, but argparse usage typically displays `-o` only.)
    python benchmarks/run.py --suite "$suite" $PYTHON_ARGS -o "$out"

    if [[ ! -s "$out" ]]; then
      echo "ERROR: expected output JSON was not created: $out" >&2
      echo "ERROR: suite=$suite label=$label PYTHON_ARGS=${PYTHON_ARGS:-<empty>}" >&2
      exit 1
    fi

    # Confirm the output metadata matches the expected arch.
    python -c "import json; m=json.load(open(\"$out\", \"r\"))[\"metadata\"]; print(\"wrote\", \"$out\", \"machine=\", m.get(\"platform.machine\"), \"python_executable=\", m.get(\"python_executable\"))"
    if [[ "$(python -c "import json; print(json.load(open(\"$out\", \"r\"))[\"metadata\"].get(\"platform.machine\", \"\"))")" != "$label" ]]; then
      echo "ERROR: $out recorded platform.machine != $label"
      exit 1
    fi
  done

  deactivate
}

IFS=',' read -r -a SUITE_ARR <<<"$SUITES"

run_in_env "$ARM_VENV" "arm64" "$ARM_PY" "${SUITE_ARR[@]}"
run_in_env "$X86_VENV" "x86_64" "$X86_PY" "${SUITE_ARR[@]}"

if [[ "$PLOT" == "1" ]]; then
  echo "==> Plot comparisons"
  for suite in "${SUITE_ARR[@]}"; do
    "$ARM_VENV/bin/python3" benchmarks/plot.py \
      "$OUT_DIR/${suite}_arm64.json" \
      "$OUT_DIR/${suite}_x86_64.json" \
      --out "$OUT_DIR/${suite}_compare.png"
  done
fi

echo "==> Done. Outputs in: $OUT_DIR"


