#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

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
"$PYTHON" -m cutsite.cli split proteins.yaml --out "$WORK/s.csv" --cert "$WORK/c.json" --require-exact
"$PYTHON" -m cutsite.cli verify "$WORK/c.json" --split "$WORK/s.csv"
if "$PYTHON" -m cutsite.cli audit proteins.yaml --split-column fold_random --no-neighbours >/dev/null; then
  echo "gate: the random split audit should have failed, and did not" >&2
  exit 1
fi

echo
echo "gate: green ($(git rev-parse --short HEAD))"
