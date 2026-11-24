from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import pytest

from cutsite import api
from cutsite.errors import PolicyError
from cutsite.evaluate import SATISFIED, VIOLATED
from cutsite.policy import Policy

from conftest import make_rows, write_csv


def _policy(path: Path, **over) -> Dict:
    base = {
        "cutsite": 1,
        "dataset": {
            "path": str(path),
            "id": "accession",
            "label": "activity",
            "similarity": {"sequence": {"column": "sequence", "kind": "protein"}},
            "groups": [],
            "dates": {"deposited": "deposited"},
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.7, "test": 0.3}},
        "constraints": [
            {"kind": "similarity", "key": "sequence", "metric": "kmer_jaccard", "threshold": 0.3}
        ],
        "seed": 7,
    }
    base.update(over)
    return base


def _straddling(tmp_path: Path) -> Path:
    rows = make_rows(families=10, members=4, seed=21)
    for row in rows:
        family = int(row["accession"][1:3])
        member = int(row["accession"][-1])
        if family < 3:
            row["deposited"] = "2024-06-15" if member % 2 == 0 else "2025-06-15"
        elif family % 2 == 0:
            row["deposited"] = "2024-06-15"
        else:
            row["deposited"] = "2025-06-15"
    return write_csv(tmp_path / "straddle.csv", rows)


def test_a_split_cutsite_built_passes_its_own_temporal_check(tmp_path: Path):
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    outcome = api.make_split(Policy.from_dict(payload))
    temporal = next(report for report in outcome.reports if report.kind == "temporal")
    assert temporal.status == SATISFIED
    assert temporal.measured["items_checked"] == len(outcome.data)
    assert temporal.measured["items_after_the_cutoff_in_train"] == 0
    assert outcome.certificate["status"] == SATISFIED


def test_a_late_item_in_train_is_the_only_temporal_violation(tmp_path: Path):
    rows = make_rows(families=8, members=2, seed=22)
    for position, row in enumerate(rows):
        row["deposited"] = "2025-06-15" if position % 2 else "2024-06-15"
        row["fold"] = "train"
    path = write_csv(tmp_path / "all-train.csv", rows)
    payload = _policy(path)
    payload["constraints"] = [{"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}]
    outcome = api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    temporal = outcome.reports[0]
    assert temporal.status == VIOLATED
    assert temporal.measured["items_after_the_cutoff_in_train"] == 8
    assert all("on or after the cutoff" in example["problem"] for example in temporal.examples)


def test_an_early_item_in_test_is_not_a_temporal_violation(tmp_path: Path):
    rows = make_rows(families=8, members=2, seed=23)
    for row in rows:
        row["deposited"] = "2024-06-15"
        row["fold"] = "test"
    path = write_csv(tmp_path / "all-early-test.csv", rows)
    payload = _policy(path)
    payload["constraints"] = [{"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}]
    outcome = api.audit(Policy.from_dict(payload), split_column="fold", skip_neighbours=True)
    assert outcome.reports[0].status == SATISFIED


@pytest.mark.parametrize(
    "scheme",
    [{"kind": "kfold", "folds": 4}, {"kind": "leave-unit-out"}],
)
def test_a_temporal_constraint_is_refused_where_it_could_not_bind(tmp_path: Path, scheme):
    payload = _policy(_straddling(tmp_path), scheme=scheme)
    payload["constraints"].append(
        {"kind": "temporal", "key": "deposited", "cutoff": "2025-01-01"}
    )
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert error.value.code == "POLICY_INVALID"
    assert "named folds" in error.value.message


def test_a_temporal_constraint_naming_an_unknown_fold_is_refused(tmp_path: Path):
    payload = _policy(_straddling(tmp_path))
    payload["constraints"].append(
        {
            "kind": "temporal",
            "key": "deposited",
            "cutoff": "2025-01-01",
            "before": "training",
        }
    )
    with pytest.raises(PolicyError) as error:
        Policy.from_dict(payload)
    assert "does not define" in error.value.message


def test_row_order_does_not_change_the_split(tmp_path: Path):
    rows = make_rows(families=20, members=4, seed=31)
    shuffled = list(rows)
    random.Random(5).shuffle(shuffled)

    def assignment_for(ordered, name):
        path = write_csv(tmp_path / name, ordered)
        outcome = api.make_split(Policy.from_dict(_policy(path)))
        return {outcome.data.ids[i]: fold for i, fold in outcome.assignment.fold_of.items()}

    assert assignment_for(rows, "plain.csv") == assignment_for(shuffled, "shuffled.csv")


def test_crossings_are_broken_down_by_fold_pair(tmp_path: Path):
    rows = make_rows(families=12, members=6, seed=32)
    for position, row in enumerate(rows):
        row["fold"] = ["train", "valid", "test"][position % 3]
    path = write_csv(tmp_path / "three.csv", rows)
    outcome = api.audit(Policy.from_dict(_policy(path)), split_column="fold", skip_neighbours=True)
    by_pair = outcome.reports[0].measured["cross_fold_pairs_by_fold_pair"]
    assert set(by_pair) == {"test|train", "test|valid", "train|valid"}
    assert sum(by_pair.values()) == outcome.reports[0].measured[
        "cross_fold_pairs_at_or_above_threshold"
    ]


def test_backend_parameters_are_part_of_the_policy_digest(tmp_path: Path):
    rows = make_rows(families=8, members=3, seed=33)
    path = write_csv(tmp_path / "k.csv", rows)
    plain = api.make_split(Policy.from_dict(_policy(path)))
    tuned = _policy(path)
    tuned["constraints"][0]["params"] = {"num_perm": 64}
    tuned["constraints"][0]["backend"] = "minhash"
    adjusted = api.make_split(Policy.from_dict(tuned))
    assert plain.certificate["policy_digest"] != adjusted.certificate["policy_digest"]
    assert adjusted.certificate["policy"]["constraints"][0]["params"] == {"num_perm": 64}


def test_an_unknown_backend_parameter_is_refused(tmp_path: Path):
    from cutsite.errors import BackendUnavailable

    rows = make_rows(families=6, members=3, seed=34)
    path = write_csv(tmp_path / "bad.csv", rows)
    payload = _policy(path)
    payload["constraints"][0]["params"] = {"nonsense": 3}
    with pytest.raises(BackendUnavailable) as error:
        api.make_split(Policy.from_dict(payload))
    assert "accepted" in error.value.details
