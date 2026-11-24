from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List

import pytest

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def mutate(sequence: str, rate: float, rng: random.Random) -> str:
    letters = list(sequence)
    for position in range(len(letters)):
        if rng.random() < rate:
            letters[position] = rng.choice(ALPHABET)
    return "".join(letters)


def make_rows(
    families: int = 12,
    members: int = 5,
    length: int = 160,
    mutation_rate: float = 0.04,
    seed: int = 11,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for family in range(families):
        seed_sequence = "".join(rng.choice(ALPHABET) for _ in range(length))
        for member in range(members):
            sequence = seed_sequence if member == 0 else mutate(seed_sequence, mutation_rate, rng)
            rows.append(
                {
                    "accession": f"F{family:02d}M{member}",
                    "sequence": sequence,
                    "donor_id": f"D{family % 6}",
                    "activity": f"{(family % 3) + rng.random() * 0.2:.4f}",
                    "deposited": f"202{4 + (family % 2)}-0{1 + (member % 8)}-15",
                    "family": f"F{family:02d}",
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> Path:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def rows() -> List[Dict[str, str]]:
    return make_rows()


@pytest.fixture
def data_file(tmp_path: Path, rows: List[Dict[str, str]]) -> Path:
    return write_csv(tmp_path / "data.csv", rows)


@pytest.fixture
def policy_dict(data_file: Path) -> Dict[str, object]:
    return {
        "splitproof": 1,
        "dataset": {
            "path": str(data_file),
            "id": "accession",
            "label": "activity",
            "similarity": {"sequence": {"column": "sequence", "kind": "protein"}},
            "groups": ["donor_id"],
            "dates": {"deposited": "deposited"},
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.7, "test": 0.3}},
        "constraints": [
            {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
        ],
        "seed": 7,
    }
