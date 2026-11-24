#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${PYTHON:-}" ]; then
  PYTHON="$PWD/.venv/bin/python"
  if [ ! -x "$PYTHON" ]; then
    echo "== bootstrapping .venv with the dev extras =="
    if command -v uv >/dev/null 2>&1; then
      uv venv -q .venv
      uv pip install -q -p .venv/bin/python -e '.[dev]'
    else
      python3 -m venv .venv
      .venv/bin/python -m pip install -q -e '.[dev]'
    fi
  fi
fi
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

echo "== lint =="
"$PYTHON" -m ruff check src tools tests

echo "== types =="
"$PYTHON" -m mypy src/splitproof

echo "== unbounded-accumulation rule =="
"$PYTHON" tools/no_unbounded_pairs.py .

echo "== NET.md conformance =="
"$PYTHON" tools/netcheck.py NET.md

echo "== tests =="
"$PYTHON" -m pytest -q

echo "== end to end =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd examples
"$PYTHON" -m splitproof.cli split proteins.yaml --out "$WORK/s.csv" --cert "$WORK/c.json" --require-exact
"$PYTHON" -m splitproof.cli verify "$WORK/c.json" --split "$WORK/s.csv"
if "$PYTHON" -m splitproof.cli audit proteins.yaml --split-column fold_random --no-neighbours >/dev/null; then
  echo "gate: the random split audit should have failed, and did not" >&2
  exit 1
fi

echo
echo "gate: green ($(git rev-parse --short HEAD))"
