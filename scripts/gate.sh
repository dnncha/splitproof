#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-full}"
case "$MODE" in
  full|--static|--inner) ;;
  *) echo "usage: scripts/gate.sh [--static|--inner]" >&2; exit 2 ;;
esac

if [ -z "${PYTHON:-}" ]; then
  PYTHON="$PWD/.venv/bin/python"
  if [ ! -x "$PYTHON" ]; then
    if command -v uv >/dev/null 2>&1; then
      uv venv -q .venv
    else
      python3 -m venv .venv
    fi
  fi
  if ! "$PYTHON" -c 'import hypothesis, mypy, pyarrow, pytest, rdkit, ruff' 2>/dev/null; then
    echo "== installing the dev, chem and parquet extras in .venv =="
    if command -v uv >/dev/null 2>&1; then
      uv pip install -q -p "$PYTHON" -e '.[dev,chem,parquet]'
    else
      "$PYTHON" -m pip install -q -e '.[dev,chem,parquet]'
    fi
  fi
fi
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

echo "== lint =="
"$PYTHON" -m ruff check src tools tests scripts

echo "== types =="
"$PYTHON" -m mypy src/splitproof

echo "== unbounded-accumulation rule =="
"$PYTHON" tools/no_unbounded_pairs.py .

echo "== NET.md conformance =="
"$PYTHON" tools/netcheck.py NET.md

if [ "$MODE" = --static ]; then
  exit 0
fi
if [ "$MODE" = --inner ]; then
  exec "$PYTHON" -m pytest -q -m "not binary"
fi

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "gate: install MMseqs2 and put mmseqs on PATH to run the identity tests" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import pyarrow, rdkit'; then
  echo "gate: install the chem and parquet extras in the gate environment" >&2
  exit 1
fi
mmseqs version

echo "== tests =="
"$PYTHON" -m pytest -o addopts= -q -rs

echo "== end to end =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd examples
"$PYTHON" -m splitproof.cli split proteins.yaml --out "$WORK/s.csv" --cert "$WORK/c.json" --require-exact
"$PYTHON" -m splitproof.cli verify "$WORK/c.json" --split "$WORK/s.csv"
audit_status=0
"$PYTHON" -m splitproof.cli audit proteins.yaml --split-column fold_random --no-neighbours --json \
  >"$WORK/audit.json" || audit_status=$?
"$PYTHON" - "$audit_status" "$WORK/audit.json" <<'PYTHON'
import json
import sys
from pathlib import Path

from splitproof.cli import EXIT_VIOLATED

path = Path(sys.argv[2])
try:
    result = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    sys.exit(f"gate: random-split audit produced no JSON verdict: {error}")
if int(sys.argv[1]) != EXIT_VIOLATED or result.get("status") != "violated" or result.get("ok") is not False:
    sys.exit(f"gate: expected a violated random-split audit, got exit {sys.argv[1]} and {result!r}")
PYTHON

echo
echo "gate: green ($(git rev-parse --short HEAD))"
