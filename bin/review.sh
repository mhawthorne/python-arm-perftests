#!/usr/bin/env bash
set -euo pipefail

# Reusable local review checks.
#
# This is intentionally fast and dependency-light:
# - bytecode compile (catches syntax/import issues)
# - pre-commit hooks (secrets scan via pinned gitleaks)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash bin/review.sh [--sync]

What it does:
  - (optional) uv sync (dev group)
  - python -m compileall (sanity)
  - pre-commit run --all-files (security scan, etc.)
EOF
}

DO_SYNC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync)
      DO_SYNC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Install uv and retry." >&2
  exit 127
fi

# Keep caches inside the repo so this works in restricted environments/CI.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
export PRE_COMMIT_HOME="${PRE_COMMIT_HOME:-$ROOT/.pre-commit-cache}"

if [[ "$DO_SYNC" == "1" ]]; then
  echo "==> Sync dev deps"
  uv sync --group dev >/dev/null
fi

echo "==> Compile (sanity)"
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="python"
fi
"$PY" -m compileall -q benchmarks

echo "==> pre-commit (all files)"
uv run pre-commit run --all-files

echo "==> OK"


