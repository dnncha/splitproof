from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from splitproof import api
from splitproof.evaluate import SATISFIED_UNPROVEN, VIOLATED
from splitproof.policy import Policy
from splitproof.sim import HEURISTIC
from splitproof.sim.mmseqs import MMseqs2Identity

from conftest import make_rows, write_csv

pytestmark = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="MMseqs2 is not installed on this machine"
)


@pytest.fixture
def identity_policy(policy_dict):
    policy_dict["constraints"][0] = {
        "kind": "similarity",
        "key": "sequence",
        "metric": "identity",
        "threshold": 0.3,
    }
    return policy_dict


def test_mmseqs_reports_itself_available_and_versioned():
    backend = MMseqs2Identity()
    assert backend.available()
    version = backend.version()
    assert version not in {"unavailable", "unknown"}


def test_mmseqs_finds_every_within_family_pair():
    rows = make_rows(families=8, members=4, length=220, mutation_rate=0.05, seed=31)
    values = [row["sequence"] for row in rows]
    result = MMseqs2Identity().pairs_above(values, "protein", 0.3)
    assert result.exactness == HEURISTIC
    assert len(result.pairs) == 8 * (4 * 3 // 2)
    for left, right, identity in result.pairs:
        assert rows[left]["family"] == rows[right]["family"]
        assert 0.0 < identity <= 1.0


def test_mmseqs_identities_are_fractions_not_percentages():
    rows = make_rows(families=4, members=3, seed=32)
    values = [row["sequence"] for row in rows]
    result = MMseqs2Identity().pairs_above(values, "protein", 0.3)
    assert result.pairs.max_similarity is not None
    assert result.pairs.max_similarity <= 1.0


def test_mmseqs_nearest_finds_the_relative_that_exists():
    rows = make_rows(families=6, members=4, seed=33)
    values = [row["sequence"] for row in rows]
    queries = [index for index, row in enumerate(rows) if row["accession"].endswith("M0")]
    targets = [index for index in range(len(rows)) if index not in queries]
    neighbours = MMseqs2Identity().nearest(values, "protein", queries, targets)
    assert any(neighbour is not None for neighbour in neighbours.values())
    for query, neighbour in neighbours.items():
        if neighbour is None:
            continue
        assert rows[neighbour.index]["family"] == rows[query]["family"]
        assert 0.0 < neighbour.similarity <= 1.0


def test_a_clean_identity_split_is_unproven_not_proven(identity_policy):
    outcome = api.make_split(Policy.from_dict(identity_policy))
    assert outcome.certificate["status"] == SATISFIED_UNPROVEN
    assert outcome.certificate["units"]["exactness"] == HEURISTIC
    assert outcome.ok


def test_identity_audit_catches_a_random_split(identity_policy, tmp_path: Path):
    rows = make_rows()
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    identity_policy["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    outcome = api.audit(Policy.from_dict(identity_policy), split_column="fold")
    report = outcome.reports[0]
    assert report.status == VIOLATED
    assert report.measured["cross_fold_pairs_at_or_above_threshold"] > 0
    assert report.measured["max_cross_fold_similarity"] > 0.5
    assert report.measured["backend"] == "mmseqs2"


def test_identity_audit_quantifies_the_leak_in_real_identity(identity_policy, tmp_path: Path):
    rows = make_rows()
    for position, row in enumerate(rows):
        row["fold"] = "train" if position % 2 == 0 else "test"
    identity_policy["dataset"]["path"] = str(write_csv(tmp_path / "leaky.csv", rows))
    outcome = api.audit(Policy.from_dict(identity_policy), split_column="fold")
    entry = next(iter(outcome.certificate["diagnostics"]["keys"].values()))
    assert entry["with_a_neighbour"] > 0
    assert entry["nearest_reference_similarity"]["min"] > 0.5
    assert entry["nearest_neighbour_baseline"]["compared"] > 0


def test_mmseqs_charges_the_pair_budget(identity_policy):
    from splitproof.sim import TooManyPairs

    rows = make_rows(families=6, members=5, seed=34)
    values = [row["sequence"] for row in rows]
    with pytest.raises(TooManyPairs):
        MMseqs2Identity().pairs_above(values, "protein", 0.3, max_pairs=5)
