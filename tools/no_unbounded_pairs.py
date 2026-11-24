#!/usr/bin/env python3
"""Ban the accumulation class that escaped once already.

Two patterns are refused anywhere under src/:

  .toarray() / .todense()   materialises a dense block whose size is the product
                            of two input dimensions, so it grows quadratically
                            on realistic data and is OOM-killed rather than
                            failing cleanly.

  .tolist() outside sim/__init__.py
                            turns a whole pair array into Python objects, about
                            fifty times its numpy size, which is how evaluate.py
                            came to sort every crossing pair in Python. Pair
                            arrays are iterated in chunks by PairSet only.

  .get(key, [x] * n)
                            builds the default on every call, so inside a loop
                            over rows it allocates n items per row: digest_for
                            was quadratic this way, minutes at 470k rows. Build
                            the default once, outside the loop.

  PairSet.from_triples / from_blocks outside sim/__init__.py
                            bypasses PairCollector, which is the only thing that
                            charges the pair budget. A backend that builds a
                            PairSet directly is unbounded again.

Run from the repo root. Exits 1 and names file:line on a hit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

DENSE = re.compile(r"\.(toarray|todense)\s*\(")
EAGER_SIZED_DEFAULT = re.compile(r"\.get\([^()]*,\s*\[[^\]]*\]\s*\*")
WHOLE_ARRAY_TO_PYTHON = re.compile(r"\.tolist\s*\(")
DIRECT_PAIRSET = re.compile(r"PairSet\.(from_triples|from_blocks)\s*\(")

COLLECTOR_HOME = Path("src/splitproof/sim/__init__.py")

ALLOW = "splitproof: allow-unbounded"


def scan(root: Path) -> List[Tuple[Path, int, str, str]]:
    findings: List[Tuple[Path, int, str, str]] = []
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if ALLOW in line:
                continue
            if DENSE.search(line):
                findings.append(
                    (
                        relative,
                        number,
                        "dense-materialisation",
                        "builds a dense block that grows with the product of two input sizes; "
                        "work on the sparse nonzeros instead",
                    )
                )
            if EAGER_SIZED_DEFAULT.search(line):
                findings.append(
                    (
                        relative,
                        number,
                        "eager-sized-default",
                        "builds an n-sized default on every call; build it once outside the loop",
                    )
                )
            if WHOLE_ARRAY_TO_PYTHON.search(line) and relative != COLLECTOR_HOME:
                findings.append(
                    (
                        relative,
                        number,
                        "array-to-python",
                        "converts a whole array to Python objects; keep pair work in numpy "
                        "(np.partition, np.lexsort, np.unique) or iterate the PairSet in chunks",
                    )
                )
            if DIRECT_PAIRSET.search(line) and relative != COLLECTOR_HOME:
                findings.append(
                    (
                        relative,
                        number,
                        "unbudgeted-pairs",
                        "builds a PairSet without PairCollector, so no pair budget is charged; "
                        "use PairCollector(name, threshold, max_pairs)",
                    )
                )
    return findings


def main(argv: List[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    findings = scan(root)
    if not findings:
        print("no_unbounded_pairs: clean")
        return 0
    for path, number, rule, message in findings:
        print(f"{path}:{number}: {rule}: {message}", file=sys.stderr)
    print(
        f"\nno_unbounded_pairs: {len(findings)} hit(s). "
        f"If a use really is bounded, justify it on the line with '{ALLOW}'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
