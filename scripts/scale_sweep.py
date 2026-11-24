"""Fit how split, audit and verify grow with row count, and fail on quadratic growth.

Every path here is meant to be close to linear in rows once similarity pairs are
excluded, so the fitted exponent between n and 4n should sit near 1. Quadratic
code gives 2. The bound of 1.5 is the midpoint between the two, not a fit to
today's timings, and the run prints every measurement either way.
"""
from __future__ import annotations

import csv
import math
import random
import sys
import tempfile
import time
from pathlib import Path

from splitproof import api, cert
from splitproof.policy import Policy

QUADRATIC_MIDPOINT = 1.5


def _dataset(directory: Path, units: int) -> Path:
    rng = random.Random(units)
    path = directory / f"rows-{units}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "grp", "when", "label", "fold"])
        row = 0
        for unit in range(units):
            for _ in range(rng.choice([1, 1, 1, 2, 3, 9])):
                writer.writerow([f"r{row}", f"g{unit}", f"20{10 + unit % 12}-0{1 + unit % 9}-1{unit % 9}", rng.choice("ab"), rng.choice(["train", "test"])])
                row += 1
    return path


def _policy(path: Path) -> Policy:
    return Policy.from_dict(
        {
            "splitproof": 1,
            "dataset": {"path": str(path), "id": "id", "groups": ["grp"], "dates": {"w": "when"}, "label": "label"},
            "scheme": {"kind": "holdout", "fractions": {"train": 0.8, "test": 0.2}},
            "constraints": [{"kind": "disjoint", "key": "grp"}],
            "seed": 1,
        }
    )


def _timed(action) -> float:
    start = time.process_time()
    action()
    return time.process_time() - start


def measure(directory: Path, units: int) -> dict:
    path = _dataset(directory, units)
    policy = _policy(path)
    outcome = None

    def split():
        nonlocal outcome
        outcome = api.make_split(policy, skip_neighbours=True)

    out = {"split": _timed(split)}
    split_path, cert_path = directory / f"s-{units}.csv", directory / f"c-{units}.json"
    api.write_split(split_path, outcome.data, outcome.assignment, "id")
    cert.write(cert_path, outcome.certificate)
    out["audit"] = _timed(lambda: api.audit(policy, split_column="fold", skip_neighbours=True))
    out["verify"] = _timed(lambda: api.verify(cert_path, data_path=path, split_path=split_path, skip_neighbours=True))
    return out


def main() -> int:
    small = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        low, high = measure(directory, small), measure(directory, small * 4)
    failed = False
    for name in low:
        exponent = math.log(max(high[name], 1e-6) / max(low[name], 1e-6)) / math.log(4)
        verdict = "ok" if exponent < QUADRATIC_MIDPOINT else "QUADRATIC"
        failed |= exponent >= QUADRATIC_MIDPOINT
        print(f"{name:7s} {low[name]:7.2f} s at {small} units, {high[name]:7.2f} s at {small * 4}: exponent {exponent:.2f} {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
