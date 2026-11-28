from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from splitproof import api
from splitproof.policy import Policy
from splitproof.sim import TooManyPairs
from splitproof.sim.kmer import MinHashJaccard, stable_hash

from conftest import make_rows, write_csv

PROBE = """
import hashlib, json, sys
sys.path.insert(0, {tests!r})
from conftest import make_rows
from splitproof.sim.kmer import MinHashJaccard
rows = make_rows(families=30, members=6, length=250, mutation_rate=0.02, seed=3)
values = [row["sequence"] for row in rows]
result = MinHashJaccard().pairs_above(values, "protein", 0.9)
edges = sorted((int(a), int(b)) for a, b, _ in result.pairs)
print(json.dumps({{"count": len(edges), "digest": hashlib.sha256(str(edges).encode()).hexdigest()}}))
"""


def _probe(hash_seed: str) -> dict:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(tests=str(Path(__file__).parent))],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode:
        pytest.fail(f"MinHash subprocess failed for PYTHONHASHSEED={hash_seed}: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_minhash_is_identical_under_different_interpreter_hash_seeds():
    """The guard for a real escape: Python's hash() is salted per process.

    MinHash used it for k-mer hashing, so at strict thresholds, where LSH recall is
    partial, the same command found different pair sets in different processes and
    therefore produced different splits.
    """
    results = [_probe(seed) for seed in ("0", "1", "12345")]
    assert len({entry["digest"] for entry in results}) == 1, results
    assert len({entry["count"] for entry in results}) == 1, results


def test_the_token_hash_does_not_depend_on_the_process():
    salt = (7).to_bytes(8, "little")
    assert stable_hash("ACDEF", salt) == stable_hash("ACDEF", salt)
    assert stable_hash("ACDEF", salt) != stable_hash("ACDEG", salt)
    assert stable_hash("ACDEF", salt) != stable_hash("ACDEF", (8).to_bytes(8, "little"))


def test_minhash_seed_changes_the_signature_but_not_reproducibility():
    rows = make_rows(families=20, members=5, seed=4)
    values = [row["sequence"] for row in rows]
    first = MinHashJaccard(seed=1).pairs_above(values, "protein", 0.9)
    again = MinHashJaccard(seed=1).pairs_above(values, "protein", 0.9)
    other = MinHashJaccard(seed=2).pairs_above(values, "protein", 0.9)
    assert [tuple(p[:2]) for p in first.pairs] == [tuple(p[:2]) for p in again.pairs]
    assert isinstance(other.pairs.max_similarity, (float, type(None)))


def test_minhash_refuses_a_quadratic_candidate_set_before_building_it():
    values = ["ACDEFGHIKLMNPQRSTVWY" * 10] * 3000
    with pytest.raises(TooManyPairs) as error:
        MinHashJaccard().pairs_above(values, "protein", 0.9, max_pairs=1000)
    assert error.value.details["stage"] == "candidate_generation"
    assert error.value.details["candidate_pairs"] > 1000


def _assignment(tmp_path: Path, rows, name: str, label: bool) -> dict:
    path = write_csv(tmp_path / name, rows)
    dataset = {
        "path": str(path),
        "id": "accession",
        "similarity": {"sequence": {"column": "sequence", "kind": "protein"}},
        "groups": [],
        "dates": {},
    }
    if label:
        dataset["label"] = "activity"
    payload = {
        "splitproof": 1,
        "dataset": dataset,
        "scheme": {"kind": "holdout", "fractions": {"train": 0.5, "test": 0.5}},
        "constraints": [
            {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
        ],
        "seed": 7,
    }
    outcome = api.make_split(Policy.from_dict(payload))
    return {outcome.data.ids[i]: fold for i, fold in outcome.assignment.fold_of.items()}


@pytest.mark.parametrize("label", [True, False])
def test_equal_units_with_no_distinguishing_feature_still_split_identically(tmp_path: Path, label):
    """Equal unit sizes and an even split leave nothing but the tiebreak to decide.

    The earlier invariance test could not fail, because unequal unit sizes made the
    allocator's objective pick the fold on its own.
    """
    rows = make_rows(families=8, members=3, seed=41)
    for row in rows:
        row["activity"] = "same"
    for attempt in range(10):
        shuffled = list(rows)
        random.Random(attempt).shuffle(shuffled)
        assert _assignment(tmp_path, rows, f"plain{attempt}.csv", label) == _assignment(
            tmp_path, shuffled, f"shuffled{attempt}.csv", label
        )


def test_the_certificate_is_byte_identical_across_runs(policy_dict, tmp_path: Path):
    def digest_of() -> str:
        certificate = api.make_split(Policy.from_dict(policy_dict)).certificate
        payload = {key: value for key, value in certificate.items() if key not in ("created", "environment")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    assert digest_of() == digest_of()


def test_minhash_refuses_before_enumerating_a_single_huge_bucket():
    """Round 2 moved the guard after the inner loop, re-opening the OOM it closed.

    One colliding bucket of m members inserts m(m-1)/2 tuples before any check, and
    MinHash is the default above 20000 rows, so a single bucket could exhaust memory
    before the budget was consulted.
    """
    values = ["ACDEFGHIKLMNPQRSTVWY" * 10] * 4000
    with pytest.raises(TooManyPairs) as error:
        MinHashJaccard().pairs_above(values, "protein", 0.9, max_pairs=1000)
    assert error.value.details["stage"] == "candidate_generation"
    assert error.value.details["candidate_pairs"] > 1000
